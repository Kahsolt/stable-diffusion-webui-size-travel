import os
import random
from traceback import print_exc
from typing import List, Tuple

import gradio as gr
import numpy as np
try: from moviepy.editor import concatenate_videoclips, ImageClip
except ImportError: print(f"moviepy python module not installed. Will not be able to generate video.")

import modules.scripts as scripts
from modules.processing import Processed, process_images, StableDiffusionProcessing, get_fixed_seed
from modules.shared import state
from modules.devices import torch_gc

DEFAULT_MODE          = 'simple'
DEFAULT_STEP          = 64
DEFAULT_SIZE          = 512
DEFAULT_VIDEO_SAVE    = True
DEFAULT_VIDEO_FPS     = 10
DEFAULT_VIDEO_CONCAT  = 'compose'
DEFAULT_DEBUG         = True

HINT_H_OPTS  = '<start>:<end>:<step>, e.g.: 512:1024:64'
HINT_W_OPTS  = '<start>:<end>:<step>, e.g.: 512:1024:64'
HINT_HW_OPTS = '<h_start>:<h_end>:<h_step>:<w_start>:<w_end>:<w_step>, e.g.: 512:768:768:512:32'


def _list_to_int(ls:List[str]):
    return [int(x.strip()) for x in ls]

def hwrange(start, end, step=DEFAULT_STEP):
    def _offset(end:int, step:int):
        if step > 0: return end + 1
        if step < 0: return end - 1
    
    assert start > 0 and end > 0, 'range boundary should be positive'
    assert step > 0, 'step size must be postive! (the ascending/descending order is auto inferred from `start` and `end`:)' 

    if start > end: step = -step
    return list(range(start, _offset(end, step), step))

def parse_simple_opts(s:str) -> List[int]:
    r = []

    sect = s.strip()        # '<start>:<end>:<step>'
    if ':' in sect:
        segs = _list_to_int(sect.split(':'))
        if len(segs) == 2:
            start, end = segs[0], segs[1]
            r.extend(hwrange(start, end))
        elif len(segs) == 3:
            start, end, step = segs[0], segs[1], segs[2]
            r.extend(hwrange(start, end, step))
        else: raise ValueError(f'unkonw format for sect {sect}')
    else:
        r.append(int(sect))

    return r

def zip_hw(heights:List[int], widths:List[int]) -> List[Tuple[int, int]]:
    if not heights or not widths: return [ ]

    maxlen = max(len(heights), len(widths))
    while len(heights) < maxlen: heights.append(heights[-1])
    while len(widths)  < maxlen: widths .append(widths[-1])

    return [(h, w) for h, w in zip(heights, widths)]

def parse_advance_opts(s:str) -> List[Tuple[int, int]]:
    r = []

    # replace -1 to current h/w
    def _(x, hw):
        if x == -1:
            if r: return r[-1][hw]
            else: return DEFAULT_SIZE
        else: return x
    def _h(x): return _(x, 0)
    def _w(x): return _(x, 1)

    def parse_1_seg(segs):
        hw, = segs
        r.append((_h(hw), _w(hw)))

    def parse_2_seg(segs):
        h, w = segs
        r.append((_h(h), _w(w)))

    def parse_3_seg(segs):
        hw_start, hw_end, step = segs
        hw_start, hw_end = _h(hw_start), _w(hw_end)
        r.extend([(hw, hw) for hw in hwrange(hw_start, hw_end, step)])

    def parse_4_seg(segs):
        h_start, h_end, w_start, w_end = segs
        h_start, h_end = _h(h_start), _w(h_end)
        w_start, w_end = _h(w_start), _w(w_end)
        hs = hwrange(h_start, h_end)
        ws = hwrange(w_start, w_end)
        hws = zip_hw(hs, ws)
        r.extend(hws)

    def parse_5_seg(segs):
        h_start, h_end, w_start, w_end, step = segs
        h_start, h_end = _h(h_start), _w(h_end)
        w_start, w_end = _h(w_start), _w(w_end)
        hs = hwrange(h_start, h_end, step)
        ws = hwrange(w_start, w_end, step)
        hws = zip_hw(hs, ws)
        r.extend(hws)

    def parse_6_seg(segs):
        h_start, h_end, h_step, w_start, w_end, w_step = segs
        h_start, h_end = _h(h_start), _w(h_end)
        w_start, w_end = _h(w_start), _w(w_end)
        hs = hwrange(h_start, h_end, h_step)
        ws = hwrange(w_start, w_end, w_step)
        hws = zip_hw(hs, ws)
        r.extend(hws)

    sects = s.strip().split(',')
    for sect in sects:        # '<h_start>:<h_end>:<h_step>:<w_start>:<w_end>:<w_step>'
        segs = _list_to_int(sect.strip().split(':'))
        locals().get(f'parse_{len(segs)}_seg')(segs)
    
    if r:       # deduplicate
        rr = [r[0]]
        for hw in r[1:]:
            if hw != rr[-1]:
                rr.append(hw)
        return rr
    else:
        return r


class Script(scripts.Script):

    def title(self):
        return 'Size Travel'

    def describe(self):
        return "Travel through a series of image sizes and generates a video."

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        mode        = gr.Radio(choices=['simple', 'advance'],                   value=lambda: DEFAULT_MODE)
        height_opt  = gr.Textbox(label='Height Variation (Simple Mode)',        lines=1, placeholder=HINT_H_OPTS)
        width_opt   = gr.Textbox(label='Width Variation (Simple Mode)',         lines=1, placeholder=HINT_W_OPTS)
        advance_opt = gr.Textbox(label='Height/Width Variation (Advance Mode)', lines=3, placeholder=HINT_HW_OPTS)

        video_save   = gr.Checkbox(label='Save results as video', value=lambda: DEFAULT_VIDEO_SAVE)
        video_fps    = gr.Number(label='Frames per second',       value=lambda: DEFAULT_VIDEO_FPS)
        video_concat = gr.Radio(choices=['compose', 'chain'],     value=lambda: DEFAULT_VIDEO_CONCAT)

        show_debug  = gr.Checkbox(label='Show verbose debug info at console', value=lambda: DEFAULT_DEBUG)

        return [mode, height_opt, width_opt, advance_opt, video_save, video_fps, video_concat, show_debug]

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

    def run(self, p:StableDiffusionProcessing, mode, height_opt, width_opt, advance_opt, video_save, video_fps, video_concat, show_debug):
        initial_info = None
        images = []

        if mode == 'simple':
            if not height_opt or not width_opt:
                print('run in simple mode but got empty "height_opt" or "width_opt"')
                return Processed(p, images, p.seed)
            
            hs = parse_simple_opts(height_opt)
            ws = parse_simple_opts(width_opt)
            hws = zip_hw(hs, ws)
        elif mode == 'advance':
            if not advance_opt:
                print('run in advance mode, but get empty "advance_opt"')
                return Processed(p, images, p.seed)

            hws = parse_advance_opts(advance_opt)
        else:
            print(f'unknown size_travel mode {mode}')
            return Processed(p, images, p.seed)

        if show_debug:
            print('[size_travel] hws:', hws)

        # Custom seed travel saving
        travel_path = os.path.join(p.outpath_samples, 'size_travel')
        os.makedirs(travel_path, exist_ok=True)
        travel_number = Script.get_next_sequence_number(travel_path)
        travel_path = os.path.join(travel_path, f"{travel_number:05}")
        p.outpath_samples = travel_path

        # Force Batch Count and Batch Size to 1.
        p.n_iter     = 1
        p.batch_size = 1

        # Random unified const seed
        seed = get_fixed_seed(p.seed)
        p.seed             = seed
        p.subseed          = None
        p.subseed_strength = 0.0
        if show_debug: print('seed:', p.seed)

        # Start job
        n_jobs = len(hws)
        state.job_count = n_jobs
        print(f"Generating {n_jobs} images.")
        for h, w in hws:
            if state.interrupted: break

            p.height = h
            p.width  = w

            try:
                proc = process_images(p)
                if initial_info is None: initial_info = proc.info
                images += proc.images
            except:
                print(f'>> error gen size ({h}, {w})')
                if show_debug: print_exc()

        if video_save:
            try:
                imgs = [np.asarray(t) for t in images]
                frames = [ImageClip(img, duration=1/video_fps) for img in imgs]
                clip = concatenate_videoclips(frames, method=video_concat)            # images may have different size
                clip.fps = video_fps
                clip.write_videofile(os.path.join(travel_path, f"travel-{travel_number:05}.mp4"), verbose=False, audio=False, logger=None)
            except:
                print_exc()

        return Processed(p, images, p.seed, initial_info)


if __name__ == '__main__':
    # simple mode
    assert parse_simple_opts('512:768:32')      == [512, 544, 576, 608, 640, 672, 704, 736, 768]
    assert parse_simple_opts('768:512:32')      == [768, 736, 704, 672, 640, 608, 576, 544, 512]
    assert parse_simple_opts('512:768')         == [512, 544, 576, 608, 640, 672, 704, 736, 768]
    assert parse_simple_opts('512')             == [512]
    assert parse_simple_opts('512:768:114514')  == [512]

    hs = parse_simple_opts('512:768:128') == [512, 640, 768]
    ws = parse_simple_opts('512')         == [512]
    assert zip_hw(hs, ws) == [(512, 512), (640, 512), (768, 512)]
    ws = parse_simple_opts('512:768:256') == [512, 768]
    assert zip_hw(hs, ws) == [(512, 512), (640, 768), (768, 768)]

    # advance mode
    hws = parse_advance_opts('512, 512:512:10, 512:512:512:512:10, 512:512:3:512:512:3')
    assert hws == [(512, 512)]

    hws = parse_advance_opts('1:9:2:6:2')
    assert hws == [(1, 2), (3, 4), (5, 6), (7, 6), (9, 6)]

    hws = parse_advance_opts('1:3:1:30:10:-10')
    assert hws == [(1, 30), (2, 20), (3, 10)]
    hws = parse_advance_opts('1:3:1:30:10:-20')
    assert hws == [(1, 30), (2, 10), (3, 10)]
    
    hws = parse_advance_opts('512, 384:384, -1:768:128, 768:512:114514, -1:768:-1:512:128')
    assert hws == [(512, 512), (384, 384), (512, 512), (640, 640), (768, 768), (768, 640), (768, 512)]

    print('All tests passed.')
