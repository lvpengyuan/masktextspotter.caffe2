import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.signal import savgol_filter

def parse_args():
    parser = argparse.ArgumentParser(
        description='Vis loss'
    )
   
    parser.add_argument(
        '--src_dir',
        dest='src_dir',
        required=True, 
        type=str
    )
    
    parser.add_argument(
        '--log_file',
        dest='log_file',
        required=True, 
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def parse_line(line, keys):
	values = {}
	line = line.strip().split('{')[-1].split('}')[0]
	parts = line.split(',')
	for part in parts:
		temp = part.split(':')
		if str(temp[0].strip()[1:-1]) in keys:
			values[str(temp[0].strip()[1:-1])] = float(temp[1])
	return values


def parse_log(log_file, keys):
	logs = {}
	for key in keys:
		logs[key] = []

	lines = open(log_file).readlines()
	for line in lines:
		line = line.strip()
		if 'json_stats:' in line:
			temp = parse_line(line, keys)
			for key in keys:
				if key == "iter":
					logs[key].append(int(temp[key]))
				else:
					logs[key].append(temp[key])
	return logs


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth



def plot_log(x, y, save_name, key):
	fig = plt.figure()
	plt.title(key)
	plt.xlabel("iter")
	plt.ylabel(key)
	if key in ["loss"]:
		plt.axis([x[0], x[-1], 0, 2.5])
	if key in ["loss_rpn_bbox_fpn2", "loss_rpn_bbox_fpn3", "loss_rpn_bbox_fpn4", "loss_rpn_bbox_fpn5", "loss_rpn_bbox_fpn6"]:
		plt.axis([x[0], x[-1], 0, 0.01])
	if key in ["loss_rpn_cls_fpn2", "loss_rpn_cls_fpn3", "loss_rpn_cls_fpn4", "loss_rpn_cls_fpn5", "loss_rpn_cls_fpn6"]:
		plt.axis([x[0], x[-1], 0, 0.02])
	if key in ["loss_char_bbox"]:
		plt.axis([x[0], x[-1], 0, 0.05])
	if key in ["loss_global_mask", "loss_char_mask"]:
		plt.axis([x[0], x[-1], 0, 0.4])
	if key in ["accuracy_cls"]:
		plt.axis([x[0], x[-1], 0.9, 1])
	if key in ["loss_char_bbox"]:
		plt.axis([x[0], x[-1], 0, 0.01])
	plt.plot(x, y, 'r-', lw=2)
	plt.plot(x, smooth(y, 20), 'g-', lw=2)
	# plt.plot(x, savgol_filter(y, 51, 3), 'g-', lw=2)

	fig.savefig(save_name)

	

def vis_log(src_dir, log_file):
	assert (os.path.exists(os.path.join(src_dir, log_file)))
	keys = ['accuracy_cls', "iter", "loss", "loss_bbox", "loss_char_mask", "loss_cls", "loss_global_mask", "loss_rpn_bbox_fpn2", "loss_rpn_bbox_fpn3", "loss_char_bbox",
	"loss_rpn_bbox_fpn4", "loss_rpn_bbox_fpn5", "loss_rpn_bbox_fpn6", "loss_rpn_cls_fpn2", "loss_rpn_cls_fpn3", "loss_rpn_cls_fpn4", "loss_rpn_cls_fpn5", "loss_rpn_cls_fpn6"]
	logs = parse_log(os.path.join(src_dir, log_file), keys)

	if os.path.exists(os.path.join(src_dir, log_file.split('.')[0])) == False:
		os.mkdir(os.path.join(src_dir, log_file.split('.')[0]))

	for key in keys:
		if key != "iter":
			save_name = os.path.join(src_dir, log_file.split('.')[0], key + '.jpg')
			plot_log(logs["iter"], logs[key], save_name, key)

def main():
	args = parse_args()
	vis_log(args.src_dir, args.log_file)


if __name__ == '__main__':
    main()
