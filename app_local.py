"""
Local Gradio demo for CodeFormer face restoration.
Run from the project root:
    python app_local.py
"""

import os
import sys
import cv2
import torch
import torch.nn.functional as F
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torchvision.transforms.functional import normalize

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import gpu_is_available, get_device
from basicsr.utils.realesrgan_utils import RealESRGANer
from basicsr.utils.registry import ARCH_REGISTRY

from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray

# ---- download missing weights -----------------------------------------------
pretrain_model_url = {
    'codeformer':  'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
    'detection':   'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth',
    'parsing':     'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth',
    'realesrgan':  'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth',
}

for name, url in pretrain_model_url.items():
    if name == 'codeformer':
        dst = 'weights/CodeFormer/codeformer.pth'
    elif name == 'realesrgan':
        dst = 'weights/realesrgan/RealESRGAN_x2plus.pth'
    else:
        dst = f'weights/facelib/{os.path.basename(url)}'
    if not os.path.exists(dst):
        load_file_from_url(url=url, model_dir=os.path.dirname(dst), progress=True)

# ---- load models ------------------------------------------------------------
device = get_device()

def set_realesrgan():
    half = True if gpu_is_available() else False
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    upsampler = RealESRGANer(
        scale=2,
        model_path='weights/realesrgan/RealESRGAN_x2plus.pth',
        model=model,
        tile=400,
        tile_pad=40,
        pre_pad=0,
        half=half,
    )
    return upsampler

upsampler = set_realesrgan()

codeformer_net = ARCH_REGISTRY.get('CodeFormer')(
    dim_embd=512,
    codebook_size=1024,
    n_head=8,
    n_layers=9,
    connect_list=['32', '64', '128', '256'],
).to(device)
ckpt = torch.load('weights/CodeFormer/codeformer.pth', map_location=device)['params_ema']
codeformer_net.load_state_dict(ckpt)
codeformer_net.eval()

os.makedirs('results/gradio', exist_ok=True)

# ---- inference function -----------------------------------------------------
def inference(image, background_enhance, face_upsample, upscale, codeformer_fidelity):
    try:
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if img is None:
            return None, None

        upscale = int(upscale)
        if upscale > 4:
            upscale = 4
        if upscale > 2 and max(img.shape[:2]) > 1000:
            upscale = 2
        if max(img.shape[:2]) > 1500:
            upscale = 1
            background_enhance = False
            face_upsample = False

        face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            use_parse=True,
            device=device,
        )

        face_helper.read_image(img)
        num_det = face_helper.get_face_landmarks_5(
            only_center_face=False, resize=640, eye_dist_threshold=5
        )
        print(f'Detected {num_det} face(s)')
        face_helper.align_warp_face()

        for cropped_face in face_helper.cropped_faces:
            cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    output = codeformer_net(cropped_face_t, w=codeformer_fidelity, adain=True)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except RuntimeError as e:
                print(f'Inference error: {e}')
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

            face_helper.add_restored_face(restored_face.astype('uint8'))

        bg_img = None
        if background_enhance and upsampler is not None:
            bg_img = upsampler.enhance(img, outscale=upscale)[0]

        face_helper.get_inverse_affine(None)
        face_upsampler = upsampler if face_upsample else None
        restored_img = face_helper.paste_faces_to_input_image(
            upsample_img=bg_img,
            draw_box=False,
            face_upsampler=face_upsampler,
        )

        save_path = 'results/gradio/out.png'
        imwrite(restored_img, save_path)
        restored_img_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
        return restored_img_rgb, save_path

    except Exception as e:
        print(f'Error: {e}')
        return None, None


# ---- Gradio UI --------------------------------------------------------------
with gr.Blocks(title='CodeFormer – Face Restoration') as demo:
    gr.Markdown(
        """
        # CodeFormer – Blind Face Restoration
        Upload a photo and adjust the controls. **Fidelity** 0 = highest quality, 1 = closest to original identity.
        """
    )

    with gr.Row():
        with gr.Column():
            input_img   = gr.Image(type='filepath', label='Input Image')
            bg_enhance  = gr.Checkbox(value=True,  label='Enhance Background (RealESRGAN)')
            face_ups    = gr.Checkbox(value=True,  label='Upsample Restored Face')
            rescale     = gr.Slider(1, 4, value=2, step=1,    label='Upscale Factor')
            fidelity    = gr.Slider(0, 1, value=0.5, step=0.01, label='Fidelity (0 = quality, 1 = identity)')
            run_btn     = gr.Button('Restore', variant='primary')

        with gr.Column():
            output_img  = gr.Image(type='numpy', label='Restored Image')
            output_file = gr.File(label='Download Result')

    run_btn.click(
        fn=inference,
        inputs=[input_img, bg_enhance, face_ups, rescale, fidelity],
        outputs=[output_img, output_file],
    )

    gr.Examples(
        examples=[
            ['inputs/cropped_faces/00.jpg', True, True, 2, 0.7],
            ['inputs/cropped_faces/01.jpg', True, True, 2, 0.5],
            ['inputs/cropped_faces/02.png', True, True, 2, 0.3],
        ],
        inputs=[input_img, bg_enhance, face_ups, rescale, fidelity],
    )

if __name__ == '__main__':
    demo.launch(share=False)
