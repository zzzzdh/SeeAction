import os

data_dir = '/home/cheer/Project/VideoCaptioning/data'
exes = ['.avi', '.flv', '.mkv', '.mov', '.mp4']

def make_dir(video_name):
  output_path = os.path.join(data_dir, 'Images', video_name)
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  return output_path
  

def extract(video_list):
  for video in video_list:
    output_path = make_dir('zoom_{:03d}'.format(i))
    cmd = 'ffmpeg -i "{}" -r 5 {}/%05d.png'.format(video, output_path)
    os.system(cmd)

def main():
  video_list = []
  img_list = os.listdir(os.path.join(data_dir, 'Images'))
  for root, dirs, files in os.walk(os.path.join(data_dir, 'Video_Only_Version', 'zoom_add')):
    for f in files:
      if os.path.splitext(f)[1] in exes:
        if os.path.basename(os.path.splitext(f)[0]) not in img_list:
          video_list.append(os.path.join(root, f))
  #print (video_list)
  extract(video_list)

if __name__ == '__main__':
  main()
