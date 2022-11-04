import os
import sys

import gradio

import modules.scripts as scripts
import gradio as gr
import math
import random
import cv2
from einops import rearrange, repeat
from modules.processing import Processed, process_images, fix_seed, slerp
from modules.shared import opts, cmd_opts, state
from modules import images, prompt_parser
import numpy as np
import modules.processing as processing
from skimage.exposure import match_histograms
from modules import devices, shared, sd_samplers
import torch
from PIL import Image
global_seeds = ''
global_seed = 0
init_latent = None
init_scale = 1.0
init_xoffset = 0.0
init_yoffset = 0.0
prev_image = None
def advanced_creator (shape, seeds, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0, p=None):
    global global_seeds

    parsed = []

    for one in global_seeds.split(","):
        parts = one.split(":")
        parsed.append((int(parts[0]), float(parts[1]) if len(parts) > 1 else 1))

    noises = list(map(lambda e: (devices.randn(e[0], shape), e[1]), parsed))
    while True:
        cur = noises[0]
        rest = noises[1:]
        if len(rest) <= 0:
            break
        noises = list(
            map(lambda r: (slerp(r[1] / (r[1] + cur[1]), cur[0], r[0]), r[1] * cur[1]), rest))

    return torch.stack([noises[0][0]]).to(shared.device)

noise_amount = 0.0
transform_zoom = 1.0
transform_xpos = 0.0
transform_ypos = 0.0
resample_interval = 0
def add_noise(sample: torch.Tensor, noise_amt: float) -> torch.Tensor:
    return sample + torch.randn(sample.shape, device=sample.device) * noise_amt

def maintain_colors(prev_img, color_match_sample, mode):
    if mode == 'Match Frame 0 RGB':
        return match_histograms(prev_img, color_match_sample, multichannel=True)
    elif mode == 'Match Frame 0 HSV':
        prev_img_hsv = cv2.cvtColor(prev_img, cv2.COLOR_RGB2HSV)
        color_match_hsv = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2HSV)
        matched_hsv = match_histograms(prev_img_hsv, color_match_hsv, multichannel=True)
        return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2RGB)
    else: # Match Frame 0 LAB
        prev_img_lab = cv2.cvtColor(prev_img, cv2.COLOR_RGB2LAB)
        color_match_lab = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2LAB)
        matched_lab = match_histograms(prev_img_lab, color_match_lab, multichannel=True)
        return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)

color_match_sample = None
transform_contrast = 1
symmetrical = False
prev_latent_cv2_img = None
current_resample_interval = 0
prev_resample_cv2_img = None

def pil_to_cv2(pil_img):
    cv2_img = np.array(pil_img)
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
    return cv2_img

def cv2_to_pil(cv2_img):
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv2_img)
    return pil_img

sd_model = None
init_latents = []
noise_latents = []
original_noise = None
original_latent = None
loop_blend = 0.0
def sgssampler(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
    self.sampler = sd_samplers.create_sampler_with_index(sd_samplers.samplers, self.sampler_index, self.sd_model)
    global global_seed
    global init_latent
    global init_scale
    global init_xoffset
    global init_yoffset
    global prev_image
    global noise_amount
    global color_match_sample
    global transform_zoom
    global transform_xpos
    global transform_ypos
    global transform_contrast
    global sd_model
    global animate_latent_trans
    global prev_latent_cv2_img
    global prev_resample_cv2_img
    global resample_interval
    global current_resample_interval
    global init_latents
    global noise_latents
    global original_noise
    global original_latent
    global loop_blend
    symmetrical = False
    sd_model = self.sd_model
    if not self.enable_hr and prev_image != None:
        x = processing.create_random_tensors([processing.opt_C, self.height // processing.opt_f, self.width // processing.opt_f], seeds=seeds, subseeds=subseeds, subseed_strength=self.subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)
        samples = self.sampler.sample(self, x, conditioning, unconditional_conditioning)
        return samples
    resample_frame = False
    if current_resample_interval > resample_interval-1:
        resample_frame = True
        current_resample_interval = 0
    blend_amount = float(current_resample_interval)/float(resample_interval)
    print(f"blendAmount { blend_amount} step {current_resample_interval} {resample_frame}")
    current_resample_interval+=1;
    init_width = self.width // processing.opt_f
    init_height = self.height // processing.opt_f

    if prev_image is not None and resample_frame:
        prev_image_latent = image_to_latent(prev_image)
        if prev_resample_cv2_img is not None:
            prev_latent_cv2_img = prev_resample_cv2_img
        prev_resample_cv2_img = sample_to_cv2(prev_image_latent)

    if init_latent == None:
        x = processing.create_random_tensors([processing.opt_C, self.firstphase_height // processing.opt_f, self.firstphase_width // processing.opt_f], seeds=[global_seed], subseeds=subseeds, subseed_strength=0, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)
        init_latent = self.sampler.sample(self, x, conditioning, unconditional_conditioning)
        init_latent = init_latent[:, :, self.truncate_y // 2:init_latent.shape[2] - self.truncate_y // 2, self.truncate_x // 2:init_latent.shape[3] - self.truncate_x // 2]

        hilatent = True
        if hilatent:
            init_latent = torch.nn.functional.interpolate(init_latent, size=(self.height // processing.opt_f, self.width // processing.opt_f), mode="bilinear")
        else:
            init_image = processing.decode_first_stage(sd_model,init_latent)
            lowres_samples = torch.clamp((init_image + 1.0) / 2.0, min=0.0, max=1.0)

            batch_images = []
            for i, x_sample in enumerate(lowres_samples):
                x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                x_sample = x_sample.astype(np.uint8)
                image = Image.fromarray(x_sample)
                image = images.resize_image(0, image, self.width, self.height)
                image = np.array(image).astype(np.float32) / 255.0
                image = np.moveaxis(image, 2, 0)
                batch_images.append(image)

            decoded_samples = torch.from_numpy(np.array(batch_images))
            decoded_samples = decoded_samples.to(shared.device)
            decoded_samples = 2. * decoded_samples - 1.

            samples = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(decoded_samples))
            init_latent = samples[0]

        prev_latent_cv2_img = sample_to_cv2(init_latent)
        if symmetrical:
            flip_img = cv2.flip(prev_latent_cv2_img, 1)
            prev_latent_cv2_img = cv2.addWeighted(prev_latent_cv2_img, .5, flip_img, .5, 0)
            translated_sample = cv2_to_sample(prev_latent_cv2_img)
            init_latent = translated_sample.to(devices.device)
        if init_scale != 1 or init_xoffset != 0 or init_yoffset != 0:
            prev_latent_cv2_img = translate(prev_latent_cv2_img, init_width,init_height , 0, init_scale, init_width*init_xoffset, init_height*init_yoffset)
            translated_sample = cv2_to_sample(prev_latent_cv2_img);
            init_latent = translated_sample.to(devices.device)
        color_match_sample = prev_latent_cv2_img.copy()
    blend_latent = False
    if animate_latent_trans:
        blended_img = prev_latent_cv2_img
        if prev_latent_cv2_img is not None:
            prev_latent_cv2_img = translate(prev_latent_cv2_img, init_width, init_height, 0, transform_zoom, transform_xpos, transform_ypos)
            blended_img = prev_latent_cv2_img
        if prev_resample_cv2_img is not None:
            prev_resample_cv2_img = translate(prev_resample_cv2_img, init_width, init_height, 0, transform_zoom, transform_xpos, transform_ypos)
            if prev_latent_cv2_img is not None:
                blended_img = cv2.addWeighted(prev_resample_cv2_img, blend_amount, prev_latent_cv2_img, (1.0 - blend_amount), 0)
        blended_img = maintain_colors(blended_img, color_match_sample, 'Match Frame 0 RGB')
        blended_img = blended_img * transform_contrast
        if blend_latent:
            init_latents.append(blended_img)
            if len(init_latents)>4:
                del init_latents[0]

        blended_sample = cv2_to_sample(blended_img)
        noise_sample = add_noise(blended_sample, noise_amount)
        init_latent = noise_sample.to(devices.device)

    if blend_latent and len(init_latents)>3:
        final_latent = init_latents[3]*.1+init_latents[2]*.2 + init_latents[1]*.3+ init_latents[0]*.4
        blended_sample = cv2_to_sample(final_latent)
        noise_sample = add_noise(blended_sample, noise_amount)
        init_latent = noise_sample.to(devices.device)

    shared.state.nextjob()
    self.sampler = sd_samplers.create_sampler_with_index(sd_samplers.samplers, self.sampler_index, self.sd_model)

    noise = processing.create_random_tensors(init_latent.shape[1:], seeds=seeds, subseeds=subseeds, subseed_strength=subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)

    if blend_latent:
        #ns = sample_to_cv2(noise)
        noise_latents.append(noise)
        if len(noise_latents)>5:
            del noise_latents[0]
        if len(noise_latents)>4:
            final_noise = noise_latents[4]*.1+noise_latents[3]*.25+noise_latents[2]*.3 + noise_latents[1]*.25+ noise_latents[0]*.1
            noise = final_noise
    # GC now before running the next img2img to prevent running out of memory
    devices.torch_gc()

    if original_noise is None:
        original_noise = noise
    if original_latent is None:
        original_latent = init_latent

    samples = self.sampler.sample_img2img(self, init_latent, noise, conditioning, unconditional_conditioning, steps=self.steps)
    #prev_image = self.sd_model.decode_first_stage(samples)
    return samples


def image_to_latent(prev_image):
    global sd_model
    image = np.array(prev_image).astype(np.float16) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = 2. * image - 1.
    image = image.to(shared.device)
    encoded = sd_model.encode_first_stage(image)
    sample = sd_model.get_first_stage_encoding(encoded)
    return sample


def latent_to_image(sample):
    global sd_model
    decoded_samples =  processing.decode_first_stage(sd_model,sample)
    lowres_samples = torch.clamp((decoded_samples + 1.0) / 2.0, min=0.0, max=1.0)
    batch_images = []
    for i, x_sample in enumerate(lowres_samples):
        x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
        x_sample = x_sample.astype(np.uint8)
        image = Image.fromarray(x_sample)
        #image = images.resize_image(0, image, self.width, self.height)
        #image = np.array(image).astype(np.float32) / 255.0
        #image = np.moveaxis(image, 2, 0)
        batch_images.append(image)
    """
    decoded_samples = torch.from_numpy(np.array(batch_images))
    decoded_samples = decoded_samples.to(shared.device)
    decoded_samples = 2. * decoded_samples - 1.

    samples = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(decoded_samples))
    
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
    x_sample = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
    img = Image.fromarray(x_sample.astype(np.uint8))
    """
    return batch_images[0]

def view_sample_step(self, latents, path_name_modifier=''):
    if self.save_sample_per_step or self.show_sample_per_step:
        samples = self.model.decode_first_stage(latents)
        if self.save_sample_per_step:
            fname = f'{path_name_modifier}_{self.step_index:05}.png'
            for i, sample in enumerate(samples):
                sample = sample.double().cpu().add(1).div(2).clamp(0, 1)
                sample = torch.tensor(np.array(sample))
                grid = make_grid(sample, 4).cpu()
                TF.to_pil_image(grid).save(os.path.join(self.paths_to_image_steps[i], fname))
        if self.show_sample_per_step:
            print(path_name_modifier)
            self.display_images(samples)
    return

def cv2_to_sample(sample: np.ndarray) -> torch.Tensor:
    sample = ((sample.astype(float) / 255.0) * 2) - 1
    sample = sample[None].transpose(0, 3, 1, 2).astype(np.float16)
    sample = torch.from_numpy(sample)
    return sample

def sample_to_cv2(sample: torch.Tensor, type=np.uint8) -> np.ndarray:
    sample_f32 = rearrange(sample.squeeze().cpu().numpy(), "c h w -> h w c").astype(np.float32)
    sample_f32 = ((sample_f32 * 0.5) + 0.5).clip(0, 1)
    sample_int8 = (sample_f32 * 255)
    return sample_int8.astype(type)

def translate(prev_img_cv2, width,height,angle,zoom,translation_x, translation_y):

    center = (width // 2, height // 2)
    trans_mat = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    rot_mat = cv2.getRotationMatrix2D(center, angle, zoom)
    trans_mat = np.vstack([trans_mat, [0,0,1]])
    rot_mat = np.vstack([rot_mat, [0,0,1]])
    xform = np.matmul(rot_mat, trans_mat)

    return cv2.warpPerspective(
        prev_img_cv2,
        xform,
        (prev_img_cv2.shape[1], prev_img_cv2.shape[0]),
        borderMode=cv2.BORDER_WRAP
    )



def anim_frame_warp_2d(prev_img_cv2, args, anim_args, keys, frame_idx):
    angle = keys.angle_series[frame_idx]
    zoom = keys.zoom_series[frame_idx]
    translation_x = keys.translation_x_series[frame_idx]
    translation_y = keys.translation_y_series[frame_idx]

    center = (args.W // 2, args.H // 2)
    trans_mat = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    rot_mat = cv2.getRotationMatrix2D(center, angle, zoom)
    trans_mat = np.vstack([trans_mat, [0,0,1]])
    rot_mat = np.vstack([rot_mat, [0,0,1]])
    if anim_args.flip_2d_perspective:
        perspective_flip_theta = keys.perspective_flip_theta_series[frame_idx]
        perspective_flip_phi = keys.perspective_flip_phi_series[frame_idx]
        perspective_flip_gamma = keys.perspective_flip_gamma_series[frame_idx]
        perspective_flip_fv = keys.perspective_flip_fv_series[frame_idx]
        M,sl = warpMatrix(args.W, args.H, perspective_flip_theta, perspective_flip_phi, perspective_flip_gamma, 1., perspective_flip_fv);
        post_trans_mat = np.float32([[1, 0, (args.W-sl)/2], [0, 1, (args.H-sl)/2]])
        post_trans_mat = np.vstack([post_trans_mat, [0,0,1]])
        bM = np.matmul(M, post_trans_mat)
        xform = np.matmul(bM, rot_mat, trans_mat)
    else:
        xform = np.matmul(rot_mat, trans_mat)

    return cv2.warpPerspective(
        prev_img_cv2,
        xform,
        (prev_img_cv2.shape[1], prev_img_cv2.shape[0]),
        borderMode=cv2.BORDER_WRAP if anim_args.border == 'wrap' else cv2.BORDER_REPLICATE
    )

class Script(scripts.Script):
    def title(self):
        return "Seed travel BPM3"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        output_path = gr.Textbox(label='Output Path', lines=1)
        dest_seed = gr.Textbox(label='Seeds (Comma separated)', lines=1)
        xoffsets = gr.Textbox(label='X offsets (Comma separated)', lines=1)
        with gradio.Row():
            userand = gr.Checkbox(label='Use random seeds', value=False)
            seedAmount = gr.Number(label='Number of seeds', value=1)
        with gradio.Row():
            frames = gr.Number(label='Frames', value=256)
            blendframes = gr.Number(label='Blend Frames', value=0)
            preview = gr.Checkbox(label='Preview', value=False)
        with gradio.Row():
            speed = gr.Number(label='Speed', value=10)
            kick = gr.Number(label='Kick', value=10)
            snare = gr.Number(label='Snare', value=15)
            hihat = gr.Number(label='High Hat', value=0)
            barHit = gr.Number(label='Bar', value=0)
        with gradio.Row():
            target_prompt = gr.Textbox(label="Snare effect", placeholder=None)
            snare_effect_max = gr.Number(label='Snare effect max', value=1.25)
        with gradio.Row():
            scale = gr.Number(label='Scale', value=1.0)
            xoffset = gr.Number(label='X Offset', value=0.0)
            yoffset = gr.Number(label='Y Offset', value=0.0)
        with gradio.Row():
            animate_trans = gr.Checkbox(label='Animate Trans', value=False)
            zoom = gr.Textbox(label='Zoom', value=1.0)
            transx = gr.Textbox(label='Translate X', value=0.0)
            transy = gr.Textbox(label='Translate Y', value=0.0)
        with gradio.Row():
            feedback_steps = gr.Number(label='Interval', value=1.0)
            noiseamt = gr.Textbox(label='Noise', value=0.0)
            animdenoise = gr.Number(label='Denoise strength', value=0.8)
            tweenframes = gr.Number(label='Tween frames', value=0)
            contrast =gr.Textbox(label='Contrast', value=1.0)

        return [userand, seedAmount, dest_seed, frames,blendframes, speed, kick, snare, hihat, barHit, preview,target_prompt,snare_effect_max,scale,xoffset,yoffset,zoom,transx,transy,noiseamt,animdenoise,contrast,xoffsets,output_path,animate_trans,feedback_steps,tweenframes]

    def get_next_sequence_number(path):
        from pathlib import Path
        """
        Determines and returns the next sequence number to use when saving an image in the specified directory.
        The sequence starts at 0.
        """
        result = -1
        dir = Path(path)
        for file in dir.iterdir():
            if not file.is_dir(): continue
            try:
                num = int(file.name)
                if num > result: result = num
            except ValueError:
                pass
        return result + 1

    def prompt_at_t(self, weight_indexes, prompt_list, t):
        return " AND ".join(
            [
                ":".join((prompt_list[index], str(weight * t)))
                for index, weight in weight_indexes
            ]
        )
    animate_latent_trans = False
    def run(self, p, userand, seed_count, dest_seed, frames,blendframes, speed, kick, snare, hihat, barHit, preview, target_prompt, snare_effect_max, scale, xoffset, yoffset,zoom,transx,transy,noiseamt,animdenoise,contrast,xoffsets,output_path,animate_trans,feedback_steps,tweenframes):

        real_creator = processing.create_random_tensors
        real_sampler = processing.StableDiffusionProcessingTxt2Img.sample
        try:
            #  processing.create_random_tensors = advanced_creator
            global init_scale
            global init_xoffset
            global init_yoffset
            global prev_image
            global noise_amount
            global transform_zoom
            global transform_xpos
            global transform_ypos
            global transform_contrast
            global global_seed
            global init_latent
            global color_match_sample
            global global_seeds
            global animate_latent_trans
            global prev_latent_cv2_img
            global resample_interval
            global loop_blend
            animate_latent_trans = animate_trans

            init_scale = scale
            init_xoffset = xoffset
            init_yoffset = yoffset
            processing.StableDiffusionProcessingTxt2Img.sample = sgssampler



            # Force Batch Count to 1.
            p.n_iter = 1

            # Custom seed travel saving
            if preview:
                main_travel_path = os.path.join(output_path, "travelPreviews")
            else:
                main_travel_path = os.path.join(output_path, "travels")
            os.makedirs(main_travel_path, exist_ok=True)
            source_prompt = p.prompt
            snare_effect_strength = 0
            if target_prompt:
                final_prompt = source_prompt+"("+target_prompt+":"+(str(snare_effect_max*snare_effect_strength))+")"
                p.prompt = final_prompt
            #res_indexes, prompt_flat_list, prompt_indexes = prompt_parser.get_multicond_prompt_list([source_prompt, target_prompt])
            #prompt_weights, target_weights = res_indexes
            #original_init_image = p.init_images
            origdenoise = p.denoising_strength
            #p.init_images = original_init_image
            seeds = []
            xoffs = []
            if xoffsets:
                for x in xoffsets.split(","):
                    xoffs.append(float(x.strip()))

            if userand:
                s = 0
                while s < seed_count:
                    seeds.append(random.randint(0, 2147483647))
                    s = s + 1
            elif dest_seed:
                seed_count = len(dest_seed.split(","))
                for x in dest_seed.split(","):
                    seeds.append(int(x.strip()))
            else:
                seed_count = 1
                if p.seed < 0:
                    p.seed = random.randint(0, 2147483647)
                seeds.append(p.seed)
            # Set generation helpers
            total_images = (int(frames) * (seed_count))
            if preview:
                total_images /= 32
            print(f"Generating {total_images} images from {len(seeds)} seeds")
            state.job_count = total_images * 0.489453125
            p.denoising_strength = origdenoise
            for i in range(len(seeds)):
                print(f"Generating {i}/{len(seeds)} seed:{seeds[i]}")
                if xoffsets:
                    init_xoffset = xoffs[i]
                p.seed = seeds[i]
                p.subseed = seeds[i]+1
                fix_seed(p)
                # We want to save seeds since they might have been altered by fix_seed()
                global_seed = p.seed
                subamount = 0
                stepAmount = speed / 500
                totalAmount = 0
                onFrame = -1
                bar = 1
                beat = 0
                processedFrame = False
                global_seeds = dest_seed
                init_latent = None
                prev_image = None
                color_match_sample = None
                first_file = None
                p.do_not_save_samples = False
                initial_info = None
                prev_latent_cv2_img = None
                prev_resample_cv2_img = None
                original_latent = None
                original_noise = None
                resample_interval = feedback_steps
                rollback = False
                images = []
                cframe = 0
                travel_number = Script.get_next_sequence_number(main_travel_path)
                travel_path = os.path.join(main_travel_path, f"{travel_number:05}")
                os.makedirs(travel_path, exist_ok=True)
                feedback_step = 0
                loop_frames = 8
                loop_blend = 0.0
                if frames > 1:
                    p.outpath_samples = travel_path
                for step in range(int(frames)):
                    if step > frames - loop_frames:
                        loop_blend = (step - (frames- loop_frames))/loop_frames
                    print(f"Generating frame {step}/{frames} blend:{loop_blend}")
                    p.prompt = source_prompt
                    currentAmount = 0.0
                    snare_effect_strength = 0
                    t = float(step)/float(frames)
                    transform_xpos = eval(transx)
                    transform_ypos = eval(transy)
                    transform_contrast = eval(contrast)
                    transform_zoom = eval(zoom)
                    noise_amount = eval(noiseamt)
                    if math.fmod(step, 8) == 0:
                        if step > 0:
                            currentAmount = stepAmount * kick
                        onFrame = 1
                        beat += 1
                    elif onFrame == 1:
                        currentAmount = stepAmount * 1  # kick * .5
                        onFrame += 1
                    elif onFrame == 2:
                        currentAmount = stepAmount * 1
                        onFrame += 1
                    elif onFrame == 3:
                        currentAmount = stepAmount * hihat
                        onFrame += 1
                    else:
                        currentAmount = stepAmount * 1
                        onFrame += 1
                    if onFrame > 8:
                        onFrame = 1
                    if beat == 2 or beat == 4:
                        if onFrame == 1:
                            currentAmount = stepAmount * snare
                            snare_effect_strength = 1
                        if onFrame == 2:
                            snare_effect_strength = .9
                        if onFrame == 3:
                            snare_effect_strength = .75
                        if onFrame == 4:
                            snare_effect_strength = .5
                    if beat > 4:
                        beat = 1
                        currentAmount += stepAmount * barHit  # add to bar
                        bar += 1
                    #    if bar == 5:
                    #        if onFrame == 1:
                    #            if beat == 1:
                    #                totalAmount = totalAmount * .6  # += stepAmount * 8  # halfway point
                    #                currentAmount = 0
                    # currentAmount += stepAmount*speed

                    if rollback:
                        if bar == 4:
                            if beat > 3:
                                if math.fmod(onFrame, 2) != 0:
                                    currentAmount = - onFrame * stepAmount * 8
                                else:
                                    currentAmount = - stepAmount * 1
                        if bar == 8:
                            if beat > 3:
                                if onFrame == 1:
                                    tsub = totalAmount - stepAmount * 4
                                    subamount = (tsub / 4)
                                    currentAmount = - subamount
                                elif math.fmod(onFrame, 2) != 0:
                                    currentAmount = - subamount
                                else:
                                    currentAmount = - stepAmount * 1

                    totalAmount += currentAmount
                    p.subseed_strength = totalAmount
                    if target_prompt:
                        final_prompt = source_prompt + "(" + target_prompt + ":" + (str(snare_effect_max * snare_effect_strength)) + ")"
                        p.prompt = final_prompt
                    # mess = "frame: " + str(step) + " bar: " + str(bar) + " beat: " + str(beat) + " current: " + str(currentAmount) + " total: " + str(totalAmount) + '\n'
                    # f.write(mess)
                    writeFrame = True
                    frameblend = False
                    if preview:
                        writeFrame = False
                        if onFrame == 1 and beat == 1:
                            writeFrame = True
                    if writeFrame:
                        #if step > frames - 32 and frames > 32:
                        #    p.do_not_save_samples = True
                        #else:
                        #    p.do_not_save_samples = False
                        if blendframes > 0:
                            p.do_not_save_samples = True
                            blendimages = []
                            finalimage = None
                            subadd = float(stepAmount)/float(blendframes+1.0)
                            for blendstep in range(int(blendframes)):
                                p.subseed_strength+= subadd * blendstep
                                print(f"blendAmount subseed { p.subseed_strength} blendstep {step}")
                                proc = process_images(p)
                                prev_image = proc.images[0]
                                if initial_info is None:
                                    initial_info = proc.info
                                cv2_current_image = np.array(proc.images[0])
                                cv2_current_image = cv2_current_image[:, :, ::-1].copy()
                                if finalimage is None:
                                    finalimage = cv2_current_image /float(blendframes)
                                else:
                                    finalimage = finalimage+ cv2_current_image /float(blendframes)
                                processedFrame = True
                            s2 = f'{step:05d}'
                            filename = s2 +".png"
                            tpath = os.path.join(travel_path,filename)
                            cv2.imwrite( tpath,finalimage)
                        elif tweenframes >0:
                            p.do_not_save_samples = True
                            proc = process_images(p)
                            cv2_current_image = np.array(proc.images[0])
                            cv2_current_image = cv2.cvtColor(cv2_current_image, cv2.COLOR_RGB2BGR)

                            dv= float(tweenframes+1)


                            smallz = pow(transform_zoom,1.0/(dv*dv))
                            smallx = transform_xpos/dv
                            smally = transform_ypos/dv


                            if prev_image is not None:
                                cv2_prev_image = np.array(prev_image)
                                cv2_prev_image = cv2.cvtColor(cv2_prev_image, cv2.COLOR_RGB2BGR)
                                for tweenframe in range(int(tweenframes)):
                                    trans_blend = float(tweenframe)/float(tweenframes)
                                    print(f"trans {smallz} {smallx} {smally}")
                                    cv2_prev_image = translate(cv2_prev_image, proc.images[0].width, proc.images[0].height, 0, smallz, smallx, smally)
                                    wimage = cv2.addWeighted(cv2_current_image,trans_blend,cv2_prev_image,(1.0-trans_blend),0)
                                    cframe = cframe+1;
                                    s2 = f'{cframe:05d}'
                                    filename = s2 +".png"
                                    tpath = os.path.join(travel_path,filename)
                                    cv2.imwrite( tpath,wimage)
                                    print(f"writing {cframe} blend {trans_blend}")
                                    # = cv2.cvtColor(wimage, cv2.COLOR_BGR2RGB)
                                    #prev_image = Image.fromarray(prev_image)
                                cframe = cframe+1;
                                s2 = f'{cframe:05d}'
                                filename = s2 +".png"
                                tpath = os.path.join(travel_path,filename)
                                print(f"writing {cframe}")
                                cv2.imwrite( tpath,cv2_current_image)
                                prev_image = proc.images[0]
                            else:
                                s2 = f'{cframe:05d}'
                                filename = s2 +".png"
                                tpath = os.path.join(travel_path,filename)
                                cv2.imwrite( tpath,cv2_current_image)
                                print(f"writing init {cframe}")
                                prev_image = proc.images[0]

                        elif frameblend and transform_contrast<1 and prev_image is not None:
                            p.do_not_save_samples = True
                            proc = process_images(p)
                            cv2_current_image = np.array(proc.images[0])
                            cv2_current_image = cv2.cvtColor(cv2_current_image, cv2.COLOR_RGB2BGR)
                            cv2_prev_image = np.array(prev_image)
                            cv2_prev_image = cv2.cvtColor(cv2_prev_image, cv2.COLOR_RGB2BGR)
                            #cv2_current_image = np.array(proc.images[0])
                            #cv2_current_image = cv2_current_image[:, :, ::-1].copy()
                            #cv2_prev_image = np.array(prev_image)
                            #cv2_prev_image = cv2_prev_image[:, :, ::-1].copy()
                            cv2_prev_image = translate(cv2_prev_image, prev_image.width, prev_image.height, 0, transform_zoom, transform_xpos, transform_ypos)
                            finalimage = cv2.addWeighted(cv2_current_image,transform_contrast,cv2_prev_image,(1.0-transform_contrast),0)
                            prev_image = cv2.cvtColor(finalimage, cv2.COLOR_BGR2RGB)
                            prev_image = Image.fromarray(prev_image)
                            s2 = f'{step:05d}'
                            filename = s2 +".png"
                            tpath = os.path.join(travel_path,filename)
                            cv2.imwrite( tpath,finalimage)
                        else:
                            proc = process_images(p)
                            prev_image = proc.images[0]
                            if initial_info is None:
                                initial_info = proc.info
                            images += proc.images
                            processedFrame = True
                        if animate_trans:
                            if feedback_step > feedback_steps-1:
                                feedback_step = 0
                                p.denoising_strength = animdenoise
                            else:
                                feedback_step += 1
                                p.denoising_strength = origdenoise
                        """
                        if first_file == None and frames > 8:
                            first_file = os.listdir(travel_path)[0]
                            first_file = os.path.join(travel_path, first_file)
                        if step > frames - 8 and frames > 8:
                            cv2_current_image = np.array(proc.images[0])
                            cv2_current_image = cv2_current_image[:, :, ::-1].copy()
                            framenum = int(frames - step-1)
                            framenumf = framenum+16
                            s1 = f'{framenumf:05d}'
                            read_file = first_file.replace("00000",s1)
                            print(f"reading {read_file}     {framenumf}   {s1}.")
                            cv2_target = cv2.imread(read_file)
                            alpha = min((framenum+1)/8,1)
                            beta = (1.0 - alpha)
                            s2 = f'{step:05d}'
                            v2_final_image = cv2.addWeighted(cv2_current_image,alpha,cv2_target,beta,0)
                            write_file = first_file.replace("00000", s2)
                            cv2.imwrite(write_file,v2_final_image)
                        """
            processed = Processed(p, images, p.seed, initial_info)
            processed.images = [processed.images[0]]
            #if processedFrame:
            #    ipath = os.path.join(travel_path, 'info.txt')
            #    f = open(ipath, 'w')
            #    f.write("seed1: " + str(seeds[0]) + " seed2: " + str(seeds[1]))
            #    f.close()
            return processed
        finally:
            processing.StableDiffusionProcessingTxt2Img.sample = real_sampler
            processing.create_random_tensors = real_creator

    def describe(self):
        return "Travel between two (or more) seeds and create a picture at each step."
