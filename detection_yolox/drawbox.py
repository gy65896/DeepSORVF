from PIL import Image, ImageDraw, ImageFont
import numpy as np


def drawbox(image, inp, font, thickness):

    for i in range(len(inp)):

        label = '{} {:.2f}'.format(inp[0][1], inp[0][2])

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)
        top    = inp[0][3][0]
        left   = inp[0][3][1]
        bottom = inp[0][3][2]
        right  = inp[0][3][3]
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])
        for i in range(thickness):
            draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline='red')
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill='red')
        draw.text(text_origin, str(label), fill=(0, 0, 0), font=font)
        del draw
    return image

            # label = '{} {:.2f}'.format(predicted_class, score)
            # draw = ImageDraw.Draw(image)
            # label_size = draw.textsize(label, font)
            # label = label.encode('utf-8')
            # #print(label, top, left, bottom, right)
            
            # if top - label_size[1] >= 0:
            #     text_origin = np.array([left, top - label_size[1]])
            # else:
            #     text_origin = np.array([left, top + 1])

            # for i in range(thickness):
            #     draw.rectangle(
            #         [left + i, top + i, right - i, bottom - i],
            #         outline=self.colors[self.class_names.index(predicted_class)])
            # draw.rectangle(
            #     [tuple(text_origin), tuple(text_origin + label_size)],
            #     fill=self.colors[self.class_names.index(predicted_class)])
            # draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            # del draw
