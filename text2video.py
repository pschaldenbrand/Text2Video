import torch
import numpy as np
import datetime

from utils import *
from clip_loss import *

from beautifier import cyclegan_model

def load_generator_model(model_type, n=1, ngf=64, h=None, w=None, pretrained_model=None):

    generate = lambda g, z : g(z)

    if model_type == 'cyclegan':
        z = torch.rand((n, 3, h, w), device=device)
        gen = cyclegan_model.netG_A
        generate = lambda g, z : (g(z)+ 1)/2
    else:
        class Nothin(nn.Module):
            def __init__(self):
                super(Nothin, self).__init__()
            def forward(self, z):
                return z
        gen = Nothin()
        z = torch.rand((n, 3, h, w), device=device)
        generate = lambda g, z : g(z)

    # for param in gen.parameters():
    #     param.requires_grad = True
    # gen.train()

    z.requires_grad = True

    for param in gen.parameters():
        param.requires_grad = False
    gen.eval()

    return gen, z, generate


# Image Augmentation Transformation
augment_trans, augment_trans_style, augment_change_clip = get_image_augmentation(True)

def alter_z_noise(z, squish=4, noise_std=1.):
    # Alter the params so the next image isn't exactly like the previous.
    with torch.no_grad():
        z /= squish
        z += torch.randn(z.shape).to(device) * noise_std
    return z

def generate_video( prompts, # List of text prompts to use to generate media
                    h=9*40,w=16*40,
                    lr=.1,
                    num_augs=4,
                    model_type='cyclegan',
                    debug=True,
                    frames_per_prompt=10, # Number of frames to dedicate to each prompt
                    first_iter=300, # Number of optimization iterations for first first frame
                    num_iter=50, # Optimization iterations for all but first frame
                    z_unchanging_weight=3, # Weight to ensure z does not change at all * l1_loss(z, z_prev)
                    z_noise_squish=4., # Amount to squish z by between frames
                    carry_over_iter=17, # Which iteration of optimization to use as the start of the next frame
                    encoding_comparison='cosine', # or "emd"
                    n_samples=1):

    start_time, all_canvases = time(), []
    all_latents = []

    gen, z_for_next_frame, generate = load_generator_model(model_type, n=n_samples, ngf=666, h=h, w=w, pretrained_model=None)

    # Optimizers
    #optim, style_optim, z_optim = torch.optim.Adam([z], lr=lr), torch.optim.RMSprop([z], lr=lr), torch.optim.Adam([z], lr=lr)

    content_loss, z_loss, styleloss_tot = 0, 0, 0
    prev_z = None
    image_features, image_features_16 = None, None
    total_chunks = (len(prompts)-1) * 2*frames_per_prompt + frames_per_prompt
    pbar = tqdm(total=total_chunks)

    cosine_dist = lambda a, b: -1 * torch.cosine_similarity(a, b, dim=1)
    encoding_compare = cosine_dist if encoding_comparison == 'cosine' else EMD
    l1_loss = nn.L1Loss()

    for prompt_ind in range(len(prompts)):
        prompt_now  = prompts[prompt_ind]
        prompt_next = prompts[prompt_ind+1] if prompt_ind < len(prompts)-1 else None

        with torch.no_grad():
            text_features_now  = model.encode_text(clip.tokenize(prompt_now).to(device))
            text_features_next = model.encode_text(clip.tokenize(prompt_next).to(device)) if prompt_next is not None else None
            text_features_now_16  = model_16.encode_text(clip.tokenize(prompt_now).to(device))
            text_features_next_16 = model_16.encode_text(clip.tokenize(prompt_next).to(device)) if prompt_next is not None else None

        tot_frames = frames_per_prompt*2 if prompt_ind < len(prompts)-1 else frames_per_prompt
        for frame in range(tot_frames):
            # Assign a weight to the current and next prompts
            weight_now = 1 - (frame/(tot_frames))
            weight_next = frame/(tot_frames)
            if prompt_ind == (len(prompts) - 1): weight_now = 1.

            # Alter the params so the next image isn't exactly like the previous.
            z = alter_z_noise(z_for_next_frame, squish=z_noise_squish, noise_std=1.)
            z.requires_grad = True

            # Optimizers
            optim, style_optim, z_optim = torch.optim.Adam([z], lr=lr), torch.optim.RMSprop([z], lr=lr), torch.optim.Adam([z], lr=lr)

            # Save features from previous frame
            prev_image_features = image_features.detach() if image_features is not None else None
            prev_image_features_16 = image_features_16.detach() if image_features_16 is not None else None

            # Run the main optimization loop
            iterations = first_iter if (prompt_ind==0 and frame==0) else num_iter
            for t in range(iterations):

                ''' Loss that goes through cyclegan '''
                # if t > (0.75*iterations):
                #     optim.zero_grad()
                #     z_optim.zero_grad()
                #     loss = 0

                #     # Full Sentence Loss
                #     im_batch = torch.cat([augment_trans(generate(gen, z)) for n in range(num_augs)])
                #     image_features = model.encode_image(im_batch)
                #     image_features_16 = model_16.encode_image(im_batch)
                #     for n in range(num_augs):
                #         loss -= torch.cosine_similarity(text_features_now, image_features[n:n+1], dim=1) * weight_now
                #         # if text_features_prev is not None: loss -= torch.cosine_similarity(text_features_prev, image_features[n:n+1], dim=1)
                #         #if text_features_next is not None: loss -= torch.cosine_similarity(text_features_next, image_features[n:n+1], dim=1) * weight_next

                #         loss -= torch.cosine_similarity(text_features_now_16, image_features_16[n:n+1], dim=1) * weight_now
                #         #if text_features_next_16 is not None: loss -= torch.cosine_similarity(text_features_next_16, image_features_16[n:n+1], dim=1) * weight_next

                #     content_loss = loss.item()
                #     loss.backward()
                #     optim.step()

                ''' Loss that just operates on z '''
                ex_freq = 2 # Alternate between two clip models for robustness
                z_optim.zero_grad()
                loss = 0
                im_batch = torch.cat([augment_trans(z) for n in range(num_augs)])
                if t % ex_freq == 0:
                    image_features_16 = model_16.encode_image(im_batch)
                else:
                    image_features = model.encode_image(im_batch)
                for n in range(num_augs):
                    # loss for clip features of z and text features (This and next prompt)
                    if t % ex_freq == 0:
                        loss += encoding_compare(text_features_now_16, image_features_16[n:n+1]) * weight_now
                        if text_features_next_16 is not None: loss += encoding_compare(text_features_next_16, image_features_16[n:n+1]) * weight_next
                    else:
                        loss += encoding_compare(text_features_now, image_features[n:n+1]) * weight_now
                        if text_features_next is not None: loss += encoding_compare(text_features_next, image_features[n:n+1]) * weight_next

                    if prev_image_features is not None:
                        # Loss to make sure that z doesn't change much
                        if t % 4 == 0:
                            loss += l1_loss(z, prev_z) * z_unchanging_weight

                loss.backward()
                z_loss = loss.item()
                z_optim.step()
                # z.data.clamp_(0,1)

                if t == carry_over_iter-1:
                    z_for_next_frame = z.detach().clone()

                # if t % 10 == 0 and debug:
                #     print(prompt_now)
                #     # print('LR', optim.param_groups[0]['lr'], '\tZL{:.3f}'.format(z_loss), '\tCL{:.3f}'.format(content_loss),
                #     #       '\tSL{:.3f}'.format(styleloss_tot), '\t{:.3f}min.'.format((time()-start_time)/60))
                #     gen.eval()
                #     for i in range(len(z)):
                #         with torch.no_grad():
                #             z_norm = z.detach().clone()#.clamp(0,1)
                #             img = generate(gen, z_norm).detach().cpu().numpy()[i]
                #             show_img(img)
                #             img = z_norm.detach().cpu().numpy()[i]
                #             show_img(img)

            prev_z = z.detach().clone()
            pbar.update(1)
            gen.eval()

            with torch.no_grad():
                if model_type=='cyclegan':
                    z_norm = z.detach().clone()#*2 - 1#.clamp(0,1)
                    # z_norm = transforms.Resize((2*h, 2*w))(z_norm) # Double the size. Hurts quality, slightly
                    img = generate(gen, z_norm).detach().cpu().numpy()[0]
                    # show_img(z.detach().cpu().numpy()[0])
                else:
                    img = generate(gen, z).detach().cpu().numpy()[0]
                img = draw_text_on_image(img, prompt_now)
                all_canvases.append(img)
                all_latents.append(z.detach().cpu().numpy()[0])
                # if frame % 4 == 0: print('Frame: ', frame), show_img(img)

    # to_gif(all_canvases, fn='/animation.gif')
    # from IPython.display import Image, display
    # ipython_img = Image(open('/animation.gif','rb').read())
    # display(ipython_img)

    # to_gif(all_canvases, fn='/content/drive/MyDrive/animations/{}.gif'.format(time()))
    if not os.path.exists('output'): os.mkdir('output')

    run_name = datetime.now().strftime("%m_%d__%H_%M_%S")
    fn = os.path.join('output','{}.mp4'.format(run_name))
    to_video(all_canvases, frame_rate=8, fn=fn)
    # to_video(all_canvases, frame_rate=8)
    return all_canvases, all_latents

#@title generate_video_wrapper
def generate_video_wrapper(prompts, frames_per_prompt=10, style_opt_iter=0, temperature=50):
    z_unchanging_weight = 4 - (temperature/100) * 4
    z_noise_squish = (temperature/100) * 4 + 2

    all_canvases = generate_video( prompts, # List of text prompts to use to generate media
                    # h=9*60,w=16*60,
                    lr=.1,
                    num_augs=4,#16,
                    debug=False,
                    frames_per_prompt=frames_per_prompt, # Number of frames to dedicate to each prompt
                    first_iter=100, # Number of optimization iterations for first first frame
                    num_iter=25, # Optimization iterations for all but first frame
                    carry_over_iter=13,
                    z_unchanging_weight=z_unchanging_weight, # Weight to ensure z does not change at all * l1_loss(z, z_prev)
                    z_noise_squish=z_noise_squish, # Amount to squish z by between frames
                    n_samples=1)
    return all_canvases

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Paint each frame of a video, and generate instruction files needed for robot painter.')

    # parser.add_argument("file", type=str, help='Path of the video or image to paint.')
    parser.add_argument('--prompts', nargs='+', type=str, help='')
    parser.add_argument('--temperature', type=float, default=30)

    args = parser.parse_args()

    all_canvases = generate_video_wrapper(args.prompts, frames_per_prompt=20, temperature=args.temperature)