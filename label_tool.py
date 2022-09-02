import os
import csv
import cv2
from glob import glob
from tqdm import tqdm
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout 
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.popup import Popup 
from kivy.core.window import Window
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.dropdown import DropDown
from kivy.uix.widget import Widget
from kivy.uix.checkbox import CheckBox 
from kivy.uix.image import Image 
from kivy.uix.textinput import TextInput
from kivy.uix.progressbar import ProgressBar 
from kivy.properties import DictProperty, NumericProperty, ListProperty

data_dir = '/home/cheer/Project/VideoCaptioning/data'

class label_info(BoxLayout): 
  video_quality = NumericProperty(1)
  label_information = DictProperty({'start_end': 0, 'verb': '', 'noun': '', 'noun_detail': '', 'adv': '', 'argv': ''})
  def __init__(self, **kwargs): 
    super(label_info, self).__init__(**kwargs) 
    self.orientation = 'vertical'
    self.spacing = 10
    self.verb_list = self.get_verb_list()
    self.noun_list = self.get_noun_list()
    self.verb = ''
    self.noun = ''
    self.start_end = 0
    quality_box = BoxLayout(orientation ='horizontal')
    start_end_box = BoxLayout(orientation ='horizontal', spacing = 10, size_hint = (1, 0.6))
    verb_box = BoxLayout(orientation ='horizontal', spacing = 10, size_hint = (1, 0.6))
    noun_box = BoxLayout(orientation ='horizontal', spacing = 10, size_hint = (1, 0.6))
    noun_detail_box = BoxLayout(orientation ='horizontal')
    adv_box = BoxLayout(orientation ='horizontal')
    argv_box = BoxLayout(orientation ='horizontal')

    ## start_end
    self.start_button = ToggleButton(text = 'Start', group = 'start_end', state = 'down')
    self.end_button = ToggleButton(text = 'End', group = 'start_end')
    self.start_button.bind(state = self.on_toggle_start)
    self.end_button.bind(state = self.on_toggle_end)
    start_end_box.add_widget(self.start_button)
    start_end_box.add_widget(self.end_button)

    ## quality
    low_quality = CheckBox(active = True, group = 'quality')
    medium_quality = CheckBox(active = False, group = 'quality')
    high_quality = CheckBox(active = False, group = 'quality')
    low_quality.bind(active = self.on_checkbox_low_quality)
    medium_quality.bind(active = self.on_checkbox_medium_quality)
    high_quality.bind(active = self.on_checkbox_high_quality)
    quality_box.add_widget(Label(text ='Quality: '))
    quality_box.add_widget(low_quality)
    quality_box.add_widget(medium_quality)
    quality_box.add_widget(high_quality)

    ## verb
    verb_dropdown = DropDown()
    for verb in self.verb_list:
      vbtn = Button(text = verb, size_hint_y = None, height = 40)
      vbtn.bind(on_release = lambda vbtn: verb_dropdown.select(vbtn.text))
      verb_dropdown.add_widget(vbtn)
    self.verb_button = Button(text = '--Select--')
    self.verb_button.bind(on_release = verb_dropdown.open)
    verb_dropdown.bind(on_select = lambda instance, x: self.set_verb(instance, x))
    verb_box.add_widget(Label(text = 'Verb class: ', size_hint = (0.5, 1)))
    verb_box.add_widget(self.verb_button)

    ## noun
    noun_dropdown = DropDown()
    for noun in self.noun_list:
      nbtn = Button(text = noun, size_hint_y = None, height = 40)
      nbtn.bind(on_release = lambda nbtn: noun_dropdown.select(nbtn.text))
      noun_dropdown.add_widget(nbtn)
    self.noun_button = Button(text = '--Select--')
    self.noun_button.bind(on_release = noun_dropdown.open)
    noun_dropdown.bind(on_select = lambda instance, x: self.set_noun(instance, x))
    noun_box.add_widget(Label(text = 'Noun class: ', size_hint = (0.5, 1)))
    noun_box.add_widget(self.noun_button)

    ## noun detail
    self.noun_detail_input = TextInput()
    noun_detail_box.add_widget(Label(text = 'Noun detail: ', size_hint = (0.5, 1)))
    noun_detail_box.add_widget(self.noun_detail_input)

    ## adv
    self.adv_text_input = TextInput()
    adv_box.add_widget(Label(text = 'Adv: ', size_hint = (0.5, 1)))
    adv_box.add_widget(self.adv_text_input)

    ## argv
    self.argv_text_input = TextInput()
    argv_box.add_widget(Label(text = 'Argv: ', size_hint = (0.5, 1)))
    argv_box.add_widget(self.argv_text_input)

    ## save
    save_button = Button(text = 'Set label', on_press = self.set_label, size_hint = (1, 0.6))

    self.add_widget(quality_box)
    self.add_widget(start_end_box)
    self.add_widget(verb_box)
    self.add_widget(noun_box)
    self.add_widget(noun_detail_box)
    self.add_widget(adv_box)
    self.add_widget(argv_box)
    self.add_widget(save_button)

  def on_toggle_start(self, instance, isState):
    if isState == 'down':
      self.start_end = 0
  def on_toggle_end(self, instance, isState):
    if isState == 'down':
      self.start_end = 1

  def on_checkbox_low_quality(self, instance, isActive):
    if isActive:
      self.video_quality = 1
  def on_checkbox_medium_quality(self, instance, isActive):
    if isActive:
      self.video_quality = 2
  def on_checkbox_high_quality(self, instance, isActive):
    if isActive:
      self.video_quality = 3

  def get_verb_list(self):
    with open(os.path.join(data_dir, 'Labels', 'verbs.txt'), 'r') as verb_file:
      verbs = verb_file.readlines()
    return [x.strip() for x in verbs]
  def get_noun_list(self):
    with open(os.path.join(data_dir, 'Labels', 'nouns.txt'), 'r') as verb_file:
      verbs = verb_file.readlines()
    return [x.strip() for x in verbs]

  def set_verb(self, instance, value):
    setattr(self.verb_button, 'text', value)
    self.verb = value
  def set_noun(self, instance, value):
    setattr(self.noun_button, 'text', value)
    self.noun = value

  def set_label(self, instance):
    self.label_information.update({'verb': self.verb, 'noun': self.noun, 'noun_detail': self.noun_detail_input.text, 'start_end': self.start_end, 'adv': self.adv_text_input.text, 'argv': self.argv_text_input.text})
    self.set_start_end_button_state(self.label_information['start_end'])
    self.adv_text_input.text = ''
    self.argv_text_input.text = ''
    self.noun_detail_input.text = ''

  def set_start_end_button_state(self, curr_state):
    if curr_state == 0:
      self.end_button.state = 'down'
      self.start_button.state = 'normal'
    else:
      self.start_button.state = 'down'
      self.end_button.state = 'normal'


class left_box(BoxLayout):
  boundary = DictProperty({'image_boundary': 1, 'video_boundary': 1})
  def __init__(self, index_info, **kwargs):
    super(left_box, self).__init__(**kwargs)
    self.orientation = 'vertical'
    self.image1_name = index_info['image1_name']
    self.image2_name = index_info['image2_name']
    self.video_name = index_info['video_name']
    self.video_len = index_info['video_len']
    self.image_len = index_info['image_len']
    self.image_index = index_info['image_index']
    self.video_index = index_info['video_index']
    self.description_list = self.get_description_list()
    self.description = TextInput(text = self.get_description(self.video_name))
    self.video_info = Label(text = 'Video name:' + self.video_name + ' ' + str(self.video_index + 1) + '/' + str(self.video_len) +  ' Image name:' + self.image1_name + ' ' + str(self.image_index + 1) + '/' + str(self.image_len - 1), size_hint = (1, 0.1))
    image_box = BoxLayout(orientation = 'horizontal', spacing = 10)
    self.image1 = Image(source = os.path.join(data_dir, 'Visualize', self.video_name, self.image1_name + '.png'), size_hint_x= 1, allow_stretch= True)
    self.image2 = Image(source = os.path.join(data_dir, 'Visualize', self.video_name, self.image2_name + '.png'), size_hint_x= 1, allow_stretch= True)
    self.progress_bar = ProgressBar(max = self.image_len - 1, value = 1, size_hint = (1, 0.1))
    image_box.add_widget(self.image1)
    image_box.add_widget(self.image2)
    self.add_widget(self.video_info)
    self.add_widget(self.progress_bar)
    self.add_widget(image_box)
    self.add_widget(self.description)

  def get_description_list(self):
    rows = []
    with open(os.path.join(data_dir, '1732_items.csv'), 'r') as csv_file:
      spamreader = csv.reader(csv_file)
      for row in spamreader:
        rows.append(row)
    return rows

  def get_description(self, video_name):
    print (video_name)
    for row in self.description_list:
      if video_name.split('_')[0] == row[1]:
        return row[3]
    return ''

  def go_next(self, index_info):
    if index_info['image_index'] < 1:
      self.boundary['image_boundary'] = 1
    elif index_info['image_index'] > index_info['image_len'] - 3:
      self.boundary['image_boundary'] = 2
    else:
      self.boundary['image_boundary'] = 0
    
    if index_info['video_index'] < 1:
      self.boundary['video_boundary'] = 1
    elif index_info['video_index'] > index_info['video_len'] - 2:
      self.boundary['video_boundary'] = 2
    else:
      self.boundary['video_boundary'] = 0

    self.video_name = index_info['video_name']
    self.image1_name = index_info['image1_name']
    self.image2_name = index_info['image2_name']
    self.video_info.text = 'Video name:' + self.video_name + ' ' + str(index_info['video_index'] + 1) + '/' + str(index_info['video_len']) +  ' Image name:' + index_info['image1_name'] + ' ' + str(index_info['image_index'] + 1) + '/' + str(index_info['image_len'] - 1)
    self.description.text = self.get_description(self.video_name)
    self.image1.source = os.path.join(data_dir, 'Visualize', self.video_name, self.image1_name + '.png')
    self.image2.source = os.path.join(data_dir, 'Visualize', self.video_name, self.image2_name + '.png')
    self.image1.reload()
    self.image2.reload()
    self.progress_bar.max = index_info['image_len'] - 1
    self.progress_bar.value = index_info['image_index'] + 1
    

class right_box(BoxLayout):
  index_info = DictProperty({'video_name': '', 'image1_name': '', 'image2_name': '', 'video_index': 0, 'image_index': 0, 'video_len': 0, 'image_len': 0})
  label_list = ListProperty([])
  def __init__(self, **kwargs):
    super(right_box, self).__init__(**kwargs)
    self.orientation = 'vertical'
    self.size_hint = (0.5, 1)
    self.spacing = 10
    self.video_list = self.get_video_list()
    self.index_info['video_len'] = len(self.video_list)
    self.index_info['video_name'] = self.video_list[0]
    self.image_list = self.get_image_list(self.index_info['video_name'])
    self.index_info['image1_name'] = self.image_list[0]
    self.index_info['image2_name'] = self.image_list[1] 
    self.index_info['image_len'] = len(self.image_list) 
    self.video_quality = 1  
    self.keyboard = Window.request_keyboard(self.keyboard_closed, self, 'text')
    self.keyboard.bind(on_key_down = self.on_keyboard_down)
    control_buttons = GridLayout(cols = 2, spacing = 10, size_hint = (1, 0.5))
    self.next_image_button = Button(text = 'Next image', on_press = self.go_next_image)
    self.previous_image_button = Button(text = 'Previous image', on_press = self.go_previous_image, disabled = True)
    self.next_video_button = Button(text = 'Next video', on_press = self.if_next_video)
    self.previous_video_button = Button(text = 'Previous video', on_press = self.if_previous_video, disabled = True)
    self.delete_button = Button(text = 'Delete last label', on_press = self.delete_last, disabled = True)
    self.save_button = Button(text = 'Save labels', on_press = self.save_data)
    self.label_info = label_info()
    self.label_info.bind(label_information = self.append_label, video_quality = self.set_video_quality)
    self.next_popup = Popup(title='Next', size_hint=(None, None), size=(300, 200))

    self.add_widget(self.label_info)
    control_buttons.add_widget(self.delete_button)
    control_buttons.add_widget(self.save_button)
    control_buttons.add_widget(self.previous_image_button)
    control_buttons.add_widget(self.next_image_button)
    control_buttons.add_widget(self.previous_video_button)
    control_buttons.add_widget(self.next_video_button)
    self.add_widget(control_buttons)

  def get_video_list(self):
    with open(os.path.join(data_dir, 'Labels', 'seen.txt'), 'r') as seen_file:
      lines = seen_file.readlines()
    exist_list = [x.split()[0] for x in lines]
    video_list = [os.path.basename(x).split('.')[0] for x in glob(os.path.join(data_dir, 'Diff_annotations', '[0-9]*')) if os.path.basename(x).split('.')[0] not in exist_list]
    video_list.sort()
    return video_list

  def get_image_list(self, video_name):
    image_list = []
    with open(os.path.join(data_dir, 'Diff_annotations', video_name + '.txt')) as diff_file:
      lines = diff_file.readlines()
    output_path = os.path.join(data_dir, 'Visualize', video_name)
    if not os.path.exists(output_path):
      os.makedirs(output_path)
      for line in tqdm(lines):
        if sum([int(x) for x in line.split()[2:]]) > 0:
          image_list.append(line.split()[0])
          image_name, simi, x1, y1, x2, y2, cx, cy = line.split()
          image_name = os.path.join(data_dir, 'Images', video_name, image_name + '.png')
          image = cv2.imread(image_name)
          image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4, 4)
          image = cv2.circle(image, (int(cx), int(cy)), 5, (0, 255, 0), 4)
          cv2.imwrite(image_name.replace('Images', 'Visualize'), image)
    else:
      for line in lines:
        if sum([int(x) for x in line.split()[2:]]) > 0:
          image_list.append(line.split()[0])
    image0 = cv2.imread(os.path.join(data_dir, 'Images', video_name, '00001.png'))
    cv2.imwrite(os.path.join(data_dir, 'Visualize', video_name, '00001.png'), image0)
    image_list.insert(0, '00001')
    return image_list

  def keyboard_closed(self):
    self.keyboard.unbind(on_key_down=self.on_keyboard_down)
    self.keyboard = None

  def on_keyboard_down(self, keyboard, keycode, text, modifiers):
    ## a, left
    if keycode[1] == 'a' or keycode[1] == 'left':
      if self.previous_image_button.disabled == False:
        new_index = {'image_index' : self.index_info['image_index'] - 1, 'image1_name': self.image_list[self.index_info['image_index'] - 1], 'image2_name': self.image_list[self.index_info['image_index']]}
        self.index_info.update(new_index)
    ## d, right
    elif keycode[1] == 'd' or keycode[1] == 'right':
      if self.next_image_button.disabled == False:
        new_index = {'image_index' : self.index_info['image_index'] + 1, 'image1_name': self.image_list[self.index_info['image_index'] + 1], 'image2_name': self.image_list[self.index_info['image_index'] + 2]}
        self.index_info.update(new_index)
    return True

  def go_next_image(self, instance):
    new_index = {'image_index' : self.index_info['image_index'] + 1, 'image1_name': self.image_list[self.index_info['image_index'] + 1], 'image2_name': self.image_list[self.index_info['image_index'] + 2]}
    self.index_info.update(new_index)

  def go_previous_image(self, instance):
    new_index = {'image_index' : self.index_info['image_index'] - 1, 'image1_name': self.image_list[self.index_info['image_index'] - 1], 'image2_name': self.image_list[self.index_info['image_index']]}
    self.index_info.update(new_index)

  def if_next_video(self, instance):
    next_popup_layout = BoxLayout(orientation = 'vertical', spacing = 10)
    button_layout = BoxLayout(orientation = 'horizontal', spacing = 10)
    no_button = Button(text = 'No')
    yes_button = Button(text = 'Yes')
    popup_label = Label(text = 'Go to next video?')
    button_layout.add_widget(no_button)
    button_layout.add_widget(yes_button)
    next_popup_layout.add_widget(popup_label)
    next_popup_layout.add_widget(button_layout)
    self.next_popup.content = next_popup_layout
    self.next_popup.open()
    no_button.bind(on_press = self.next_popup.dismiss)
    yes_button.bind(on_press = self.go_next_video)

  def if_previous_video(self, instance):
    pre_popup_layout = BoxLayout(orientation = 'vertical', spacing = 10)
    button_layout = BoxLayout(orientation = 'horizontal', spacing = 10)
    no_button = Button(text = 'No')
    yes_button = Button(text = 'Yes')
    popup_label = Label(text = 'Go to previous video?')
    button_layout.add_widget(no_button)
    button_layout.add_widget(yes_button)
    pre_popup_layout.add_widget(popup_label)
    pre_popup_layout.add_widget(button_layout)
    self.next_popup.content = pre_popup_layout
    self.next_popup.open()
    no_button.bind(on_press = self.next_popup.dismiss)
    yes_button.bind(on_press = self.go_previous_video)

  def go_next_video(self, instance):
    self.next_popup.dismiss()
    self.label_list = []
    next_video_index = self.index_info['video_index'] + 1
    self.image_list = self.get_image_list(self.video_list[next_video_index])
    while len(self.image_list) < 4:
      self.seen_video(self.video_list[next_video_index], 1)
      print ('skip', self.video_list[next_video_index], len(self.image_list))
      next_video_index += 1
      self.image_list = self.get_image_list(self.video_list[next_video_index])
    new_index = {'video_index': next_video_index, 'video_name': self.video_list[next_video_index], 'image_index': 0, 'image1_name': self.image_list[0], 'image2_name': self.image_list[1], 'image_len': len(self.image_list)}
    self.index_info.update(new_index)

  def go_previous_video(self, instance):
    self.next_popup.dismiss()
    self.label_list = []
    pre_video_index = self.index_info['video_index'] - 1
    self.image_list = self.get_image_list(self.video_list[pre_video_index])
    while len(self.image_list) < 4:
      print ('skip', self.video_list[pre_video_index], len(self.image_list))
      pre_video_index -= 1
      self.image_list = self.get_image_list(self.video_list[pre_video_index])
    new_index = {'video_index': pre_video_index, 'video_name': self.video_list[pre_video_index], 'image_index': 0, 'image1_name': self.image_list[0], 'image2_name': self.image_list[1], 'image_len': len(self.image_list)}
    self.index_info.update(new_index)

  def disable_next_button(self, boundary):
    if boundary['image_boundary'] == 1:
      self.previous_image_button.disabled = True
      self.next_image_button.disabled = False
    elif boundary['image_boundary'] == 2:
      self.next_image_button.disabled = True
      self.previous_image_button.disabled = False
    else:
      self.previous_image_button.disabled = False
      self.next_image_button.disabled = False

    if boundary['video_boundary'] == 1:
      self.previous_video_button.disabled = True
      self.next_video_button.disabled = False
    elif boundary['video_boundary'] == 2:
      self.next_video_button.disabled = True
      self.previous_video_button.disabled = False
    else:
      self.previous_video_button.disabled = False
      self.next_video_button.disabled = False

  def append_label(self, instance, label_information):
    full_label = label_information.copy()
    append_popup_layout = BoxLayout(orientation = 'vertical', spacing = 10)
    close_button = Button(text = 'Close')
    popup_label = Label(text = 'Data error, please check!')
    append_popup_layout.add_widget(popup_label)
    append_popup_layout.add_widget(close_button)
    append_popup = Popup(title='Set label', content = append_popup_layout, size_hint=(None, None), size=(300, 200))
    close_button.bind(on_press = append_popup.dismiss)
    if len(self.label_list):
      last_label = self.label_list[-1]
      if last_label['start_end'] == 0 and label_information['start_end'] == 1:
        last_label.update({'end': self.index_info['image2_name'], 'start_end': 1})
        self.label_list[-1] = last_label
      elif last_label['start_end'] == 1 and label_information['start_end'] == 0:
        full_label.update({'start': self.index_info['image1_name']})
        self.label_list.append(full_label)
        self.delete_button.disabled = False
      else:
        append_popup.open()
    elif label_information['start_end'] == 0:
      full_label.update({'start': self.index_info['image1_name']})
      self.label_list.append(full_label)
      self.delete_button.disabled = False
    else:
      append_popup.open()

  def set_video_quality(self, instance, video_quality):
    self.video_quality = video_quality

  def seen_video(self, video_name, video_quality):
    with open(os.path.join(data_dir, 'Labels', 'seen.txt'), 'r') as seen_file:
      lines = seen_file.readlines()
    i = 0
    while i < len(lines):
      if video_name == lines[i].split()[0]:
        del lines[i]
      else:
        i += 1
    lines.append(video_name + ' ' + str(video_quality) + '\n')
    with open(os.path.join(data_dir, 'Labels', 'seen.txt'), 'w') as seen_file:
      seen_file.writelines(lines)

  def save_data(self, instance):
    save_popup_layout = BoxLayout(orientation = 'vertical', spacing = 10)
    close_button = Button(text = 'Close')
    if len(self.label_list) == 0:
      popup_label = Label(text = 'No data, skip video')
      self.seen_video(self.video_list[self.index_info['video_index']], self.video_quality)
    elif self.label_list[-1]['start_end']:
      popup_label = Label(text = 'Save data successfully!')
      self.seen_video(self.index_info['video_name'], self.video_quality)
      with open(os.path.join(data_dir, 'Labels', 'captions', self.index_info['video_name'] + '.csv'), 'w') as csv_file:
        label_names = ['start', 'end', 'verb', 'noun', 'noun_detail', 'adv', 'argv', 'start_end',]
        writer = csv.DictWriter(csv_file, fieldnames = label_names, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for label in self.label_list:
          writer.writerow(label)
    else:
      popup_label = Label(text = 'Data error, please check!')

    save_popup_layout.add_widget(popup_label)
    save_popup_layout.add_widget(close_button)
    save_popup = Popup(title='Save data', content = save_popup_layout, size_hint=(None, None), size=(300, 200))
    save_popup.open()
    close_button.bind(on_press = save_popup.dismiss)

  def delete_last(self, instance):
    if len(self.label_list):
      self.label_list.pop()
    if len(self.label_list) == 0:
      self.delete_button.disabled = True

class display_box(BoxLayout):
  def __init__(self, **kwargs):  
    super(display_box, self).__init__(**kwargs)
    self.orientation = 'vertical'
    self.size_hint = (0.3, 1)
    self.label_num = Label(text = '0', size_hint = (1, 0.1))
    self.label_details = TextInput()
    self.add_widget(self.label_num)
    self.add_widget(self.label_details)

  def update_text(self, label_list):
    detail_text = ''
    start_num = 0
    end_num = 0
    for label in label_list:
      if 'end' in label.keys():
        detail_text = detail_text + label['start'] + ' ' + label['end'] + ' ' + label['verb'] + ' ' + label['noun'] + ' ' + label['noun_detail'] + ' ' + label['adv'] + ' ' + label['argv'] + '\n'
      else:
        detail_text = detail_text + label['start'] + ' ' + label['verb'] + ' ' + label['noun'] + ' ' + label['noun_detail'] + ' ' + label['adv'] + ' ' + label['argv'] + '\n'
      if label['start_end'] == 0:
        start_num += 1
      else: 
        end_num += 1
    self.label_details.text = detail_text
    self.label_num.text = 'Label number:' + str(len(label_list)) + '\nStart num:' + str(start_num) + ' End num:' +  str(end_num)
     

class MainWindow(BoxLayout):
  def __init__(self, **kwargs):  
    super(MainWindow, self).__init__(**kwargs) 
    self.orientation ='horizontal'
    self.spacing = 10
    self.rightbox = right_box()
    self.rightbox.bind(index_info = self.go_next, label_list = self.display_label)
    self.leftbox = left_box(self.rightbox.index_info)
    self.leftbox.bind(boundary = self.disable_next_button)
    self.displaybox = display_box()
    self.add_widget(self.leftbox)
    self.add_widget(self.rightbox)
    self.add_widget(self.displaybox)

  def go_next(self, instance, index_info):
    self.leftbox.go_next(index_info)

  def disable_next_button(self, instance, boundary):
    self.rightbox.disable_next_button(boundary)

  def display_label(self, instance, label_list):
    self.displaybox.update_text(label_list)
  

class Label_app(App):
  def build(self):
    return MainWindow()

if __name__ == '__main__':
  Label_app().run()
