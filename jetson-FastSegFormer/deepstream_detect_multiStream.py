import sys
from syslog import LOG_WARNING

import gi
import configparser

gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from ctypes import *
import sys
import math
from utils.utils import is_aarch64
from utils.utils import bus_call
from utils.utils import PERF_DATA
import numpy as np
import pyds
import cv2
import os
import os.path
from os import path
import argparse

perf_data = None


MAX_DISPLAY_LEN = 32                          # 最大显示长度
MUXER_OUTPUT_WIDTH = 640                      # 复用器输出的宽度
MUXER_OUTPUT_HEIGHT = 480                     # 复用器输出的高度
MUXER_BATCH_TIMEOUT_USEC = 4000000            # 复用器批处理的超时时间 4000000 微秒
TILED_OUTPUT_WIDTH = 640                      # 平铺输出的宽度
TILED_OUTPUT_HEIGHT = 480                     # 平铺输出的高度
GST_CAPS_FEATURES_NVMM = "memory:NVMM"        # GStreamer流媒体框架中的一种特性，使用NVMM（NVIDIA Multimedia）内存
COLORS = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                        (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                        (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                        (128, 64, 12)]



def map_mask_as_display_bgr(mask):
    """ Assigning multiple colors as image output using the information
        contained in mask. (BGR is opencv standard.)
    """
    # getting a list of available classes
    m_list = list(set(mask.flatten()))

    shp = mask.shape
    bgr = np.zeros((shp[0], shp[1], 3))
    for idx in m_list:
        bgr[mask == idx] = COLORS[idx]
    return bgr





# Tiler_sink_pad_buffer_probe将提取在tiller sink pad上收到的元数据
# 重新调整并将二进制分割掩码数组以保存到图像
def tiler_sink_pad_buffer_probe(pad, info, u_data):
    frame_number = 0
    num_rects = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        l_obj = frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta
        is_first_obj = True
        save_image = False
        obj_number = 0
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            if is_first_obj and frame_number % 30 == 0:
                is_first_obj = False
                rectparams = obj_meta.rect_params # Retrieve rectparams for re-sizing mask to correct dims
                maskparams = obj_meta.mask_params # Retrieve maskparams
                mask_image = resize_mask(maskparams, math.floor(rectparams.width), math.floor(rectparams.height)) # Get resized mask array
                
                img_path = "{}/stream_{}/frame_{}.jpg".format(folder_name, frame_meta.pad_index, frame_number)
                cv2.imwrite(img_path, mask_image) # Save mask to image
            
            try:
                l_obj = l_obj.next
                obj_number += 1
            except StopIteration:
                break

        print("Frame Number=", frame_number, "Number of Objects=", num_rects)
        # update frame rate through this probe
        stream_index = "stream{0}".format(frame_meta.pad_index)
        global perf_data
        perf_data.update_fps(stream_index)
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def clip(val, low, high):
    if val < low:
        return low 
    elif val > high:
        return high 
    else:
        return val


# 将输入的二进制掩码数组按照指定的宽度和高度进行缩放。
def resize_mask(maskparams, target_width, target_height):
    src = maskparams.get_mask_array() # Retrieve mask array
    dst = np.empty((target_height, target_width), src.dtype) # Initialize array to store re-sized mask
    original_width = maskparams.width
    original_height = maskparams.height
    ratio_h = float(original_height) / float(target_height)
    ratio_w = float(original_width) / float(target_width)
    threshold = maskparams.threshold
    channel = 1

    # Resize from original width/height to target width/height 
    for y in range(target_height):
        for x in range(target_width):
            x0 = float(x) * ratio_w
            y0 = float(y) * ratio_h
            left = int(clip(math.floor(x0), 0.0, float(original_width - 1.0)))
            top = int(clip(math.floor(y0), 0.0, float(original_height - 1.0)))
            right = int(clip(math.ceil(x0), 0.0, float(original_width - 1.0)))
            bottom = int(clip(math.ceil(y0), 0.0, float(original_height - 1.0)))

            for c in range(channel):
                # H, W, C ordering
                # Note: lerp is shorthand for linear interpolation
                left_top_val = float(src[top * (original_width * channel) + left * (channel) + c])
                right_top_val = float(src[top * (original_width * channel) + right * (channel) + c])
                left_bottom_val = float(src[bottom * (original_width * channel) + left * (channel) + c])
                right_bottom_val = float(src[bottom * (original_width * channel) + right * (channel) + c])
                top_lerp = left_top_val + (right_top_val - left_top_val) * (x0 - left)
                bottom_lerp = left_bottom_val + (right_bottom_val - left_bottom_val) * (x0 - left)
                lerp = top_lerp + (bottom_lerp - top_lerp) * (y0 - top)
                if (lerp < threshold): # Binarize according to threshold
                    dst[y,x] = 0
                else:
                    dst[y,x] = 255
    return dst


# 回调函数，用于处理新的解码器源pad的创建。当decodebin创建一个新的解码器源pad时，会调用这个函数。
def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    if (gstname.find("video") != -1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)

    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element.find_property('drop-on-latency') != None:
            Object.set_property("drop-on-latency", True)


# 读取传入的uri参数指定的媒体文件，并自动选择合适的解码器进行解码
def create_source_bin(index, uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-%02d" % index
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin



def create_element(factoryname, name):
    element = Gst.ElementFactory.make(factoryname, name)
    if not element:
        print(f"Failed to create {factoryname}")
        sys.exit(1)
    return element


def main(stream_paths, output_folder):
    global perf_data
    perf_data = PERF_DATA(len(stream_paths))
    number_sources = len(stream_paths)

    global folder_name
    folder_name = output_folder
    if path.exists(folder_name):
        sys.stderr.write("The output folder %s already exists. Please remove it first.\n" % folder_name)
        sys.exit(1)

    os.mkdir(folder_name)
    print("Frames will be saved in ", folder_name)
    

    # 建立Pipeline
    Gst.init(None)
    print("Creating Pipeline\n")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    print("Creating streammux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)

    for i in range(number_sources):
        os.mkdir(folder_name + "/stream_" + str(i))
        print("Creating source_bin ", i, " \n ")
        uri_name = stream_paths[i]
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)
    
    print("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")
    print("Creating tiler \n ")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")
    print("Creating nvvidconv \n ")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")
    print("Creating nvosd \n ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
    if is_aarch64():
        print("Creating nv3dsink \n")
        sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
        if not sink:
            sys.stderr.write(" Unable to create nv3dsink \n")
    else:
        print("Creating EGLSink \n")
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        if not sink:
            sys.stderr.write(" Unable to create egl sink \n")

    if is_live:
        print("Atleast one of the sources is live")
        streammux.set_property('live-source', 1)
    
    streammux.set_property('width', 1280)
    streammux.set_property('height', 720)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', 4000000)
    pgie.set_property('config-file-path', "FastSegFormer_deepstream_config.txt")
    pgie_batch_size = pgie.get_property("batch-size")
    
    # 每一个输入视频流就用一个Batch
    # if (pgie_batch_size != number_sources):
    #     print("WARNING: Overriding infer-config batch-size", pgie_batch_size, " with number of sources ",
    #           number_sources, " \n")
    #     pgie.set_property("batch-size", number_sources)
    
    # tiler.set_property()函数用于设置视频流分割器的属性具体而言。
    tiler_rows = int(math.sqrt(number_sources))                                 
    tiler_columns = int(math.ceil((1.0 * number_sources) / tiler_rows))         
    tiler.set_property("rows", tiler_rows)                                      # 视频流分割器的行数
    tiler.set_property("columns", tiler_columns)                                # 视频流分割器的列数
    tiler.set_property("width", TILED_OUTPUT_WIDTH)                             # 视频流分割器的宽度
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)                           # 视频流分割器的高度
    

    nvosd.set_property("display_mask", True) # Note: display-mask is supported only for process-mode=0 (CPU) 
    nvosd.set_property('process_mode', 0)    # 显示屏幕输出模式

    sink.set_property("sync", 0)             # 该元素不需要等待时钟同步
    sink.set_property("qos", 0)              # 该元素不需要使用质量保证机制

    # 建立5个队列对视频流进行缓冲
    queue1=Gst.ElementFactory.make("queue","queue1")
    queue2=Gst.ElementFactory.make("queue","queue2")
    queue3=Gst.ElementFactory.make("queue","queue3")
    queue4=Gst.ElementFactory.make("queue","queue4")
    queue5=Gst.ElementFactory.make("queue","queue5")
    pipeline.add(queue1)
    pipeline.add(queue2)
    pipeline.add(queue3)
    pipeline.add(queue4)
    pipeline.add(queue5)

    print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)

    print("Linking elements in the Pipeline \n")
    streammux.link(queue1)
    queue1.link(pgie)
    pgie.link(queue2)
    queue2.link(tiler)
    tiler.link(queue3)
    queue3.link(nvvidconv)
    nvvidconv.link(queue4)
    queue4.link(nvosd)
    nvosd.link(queue5)
    queue5.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    tiler_sink_pad = tiler.get_static_pad("sink")
    if not tiler_sink_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        tiler_sink_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_sink_pad_buffer_probe, 0)
        # 每5秒打印一次fps
        GLib.timeout_add(5000, perf_data.perf_print_callback)

    # 列出视频流来源
    print("Now playing...")
    for i, source in enumerate(stream_paths):
        print(i, ": ", source)

    print("Starting pipeline \n")
    # start play back and listed to events		
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)



def parse_args():
    parser = argparse.ArgumentParser(prog="deepstream_detect.py", 
                description="deepstream-segmask takes multiple URI streams or single stream as input" \
                    " and re-sizes and binarizes segmentation mask arrays to save to image")
    parser.add_argument(
        "-i",
        "--input",
        help="Path to input streams",
        nargs="+",
        metavar="URIs",
        default=["a"],
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="output_folder_name",
        default="out",
        help="Name of folder to output mask images",
    )

    args = parser.parse_args()
    stream_paths = args.input
    output_folder = args.output
    return stream_paths, output_folder

    

if __name__ == '__main__':
    stream_paths, output_folder = parse_args()
    sys.exit(main(stream_paths, output_folder))

