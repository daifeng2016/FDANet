import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as F
from PIL import Image


def transforms_for_noise(inputs_u2, std):

    gaussian = np.random.normal(0, std, (inputs_u2.shape[0], 3, inputs_u2.shape[-1], inputs_u2.shape[-1]))
    gaussian = torch.from_numpy(gaussian).float().cuda()
    inputs_u2_noise = inputs_u2 + gaussian

    return inputs_u2_noise

def transforms_for_rot(ema_inputs):

    rot_mask = np.random.randint(0, 4, ema_inputs.shape[0])#ema_inputs=[N,C,H,W]
    flip_mask = np.random.randint(0, 2, ema_inputs.shape[0])

    # flip_mask = [0,0,0,0,1,1,1,1]
    # rot_mask = [0,1,2,3,0,1,2,3]

    for idx in range(ema_inputs.shape[0]):
        if flip_mask[idx] == 1:
            ema_inputs[idx] = torch.flip(ema_inputs[idx], [1])##flipup

        ema_inputs[idx] = torch.rot90(ema_inputs[idx], int(rot_mask[idx]), dims=[1,2])

    return ema_inputs, rot_mask, flip_mask


def transforms_for_scale(ema_inputs, image_size):

    scale_mask = np.random.uniform(low=0.8, high=1.2, size=ema_inputs.shape[0])
    scale_mask = scale_mask * image_size
    scale_mask = [int(item) for item in scale_mask]
    scale_mask = [item-1 if item % 2 != 0 else item for item in scale_mask]
    half_size = int(image_size / 2)

    ema_outputs = torch.zeros_like(ema_inputs)

    for idx in range(ema_inputs.shape[0]):
        # to numpy
        img = np.transpose(ema_inputs[idx].cpu().numpy(), (1,2,0))
        # crop
        if scale_mask[idx] > image_size:
            # new_img = np.zeros((scale_mask[idx], scale_mask[idx], 3), dtype="uint8")
            # new_img[int(scale_mask[idx]/2)-half_size:int(scale_mask[idx]/2)+half_size,
            # int(scale_mask[idx] / 2) - half_size:int(scale_mask[idx]/2) + half_size, :] = img
            new_img1 = np.expand_dims(np.pad(img[:, :, 0],
                                             (int((scale_mask[idx]-image_size)/2),
                                             int((scale_mask[idx]-image_size)/2)), 'edge'), axis=-1)
            new_img2 = np.expand_dims(np.pad(img[:, :, 1],
                                             (int((scale_mask[idx]-image_size)/2),
                                             int((scale_mask[idx]-image_size)/2)), 'edge'), axis=-1)
            new_img3 = np.expand_dims(np.pad(img[:, :, 2],
                                             (int((scale_mask[idx] - image_size) / 2),
                                              int((scale_mask[idx] - image_size) / 2)), 'edge'), axis=-1)
            new_img = np.concatenate([new_img1, new_img2, new_img3], axis=-1)
            img = new_img
        else:
            img = img[half_size-int(scale_mask[idx]/2):half_size + int(scale_mask[idx]/2),
            half_size-int(scale_mask[idx]/2): half_size + int(scale_mask[idx]/2),:]

        # resize
        resized_img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        # to tensor
        ema_outputs[idx] = torch.from_numpy(resized_img.transpose((2, 0, 1))).cuda()

    return ema_outputs, scale_mask



def transforms_back_scale(ema_inputs, scale_mask, image_size):
    half_size = int(image_size/2)
    returned_img = np.zeros((ema_inputs.shape[0],  image_size, image_size, ema_inputs.shape[1]))#[16,128,128,1]

    ema_outputs = torch.zeros_like(ema_inputs)#[16,1,128,128]

    for idx in range(ema_inputs.shape[0]):
        # to numpy
        img = np.transpose(ema_inputs[idx].cpu().numpy(), (1,2,0))#[128,128,1]
        # resize
        resized_img = cv2.resize(img, (int(scale_mask[idx]), int(scale_mask[idx])), interpolation=cv2.INTER_CUBIC)#
        if img.shape[-1]==1:
            resized_img=np.expand_dims(resized_img,axis=-1)

        if scale_mask[idx] > image_size:
            returned_img[idx] = resized_img[int(scale_mask[idx]/2)-half_size:int(scale_mask[idx]/2)+half_size,
            int(scale_mask[idx] / 2) - half_size:int(scale_mask[idx]/2) + half_size, :]

        else:
            returned_img[idx, half_size-int(scale_mask[idx]/2):half_size + int(scale_mask[idx]/2),
            half_size-int(scale_mask[idx]/2): half_size + int(scale_mask[idx]/2), :] = resized_img
        # to tensor
        ema_outputs[idx] = torch.from_numpy(returned_img[idx].transpose((2,0,1))).cuda()

    return ema_outputs, scale_mask

def postprocess_scale(input, scale_mask, image_size):
    half_size = int(input.shape[-1]/2)
    new_input = torch.zeros((input.shape[0], input.shape[1], input.shape[-1], input.shape[-1]))

    for idx in range(input.shape[0]):

        if scale_mask[idx] > image_size:
            new_input = input
        #     scale_num = int((image_size/(scale_mask[idx]/image_size))/2)
        #     new_input[idx, :, half_size - scale_num:half_size + scale_num,
        #     half_size - scale_num: half_size + scale_num] \
        #         = input[idx, :, half_size - scale_num:half_size + scale_num,
        #     half_size - scale_num: half_size + scale_num]
        else:
            new_input[idx, :, half_size-int(scale_mask[idx]/2):half_size + int(scale_mask[idx]/2),
            half_size-int(scale_mask[idx]/2): half_size + int(scale_mask[idx]/2)] \
            = input[idx, :, half_size-int(scale_mask[idx]/2):half_size + int(scale_mask[idx]/2),
            half_size-int(scale_mask[idx]/2): half_size + int(scale_mask[idx]/2)]

    return new_input.cuda()

def transforms_back_rot(ema_output,rot_mask, flip_mask):

    for idx in range(ema_output.shape[0]):

        ema_output[idx] = torch.rot90(ema_output[idx], int(rot_mask[idx]), dims=[2,1])

        if flip_mask[idx] == 1:
            ema_output[idx] = torch.flip(ema_output[idx], [1])

    return ema_output

def transforms_input_for_shift(ema_inputs, image_size):

    scale_mask = np.random.uniform(low=0.9, high=0.99, size=ema_inputs.shape[0])
    scale_mask = scale_mask * image_size
    scale_mask = [int(abs(item-image_size)) for item in scale_mask]
    scale_mask = [item-1 if item % 2 != 0 else item for item in scale_mask]
    scale_mask = [2 if item == 0 else item for item in scale_mask]
    # half_size = int(image_size / 2)

    shift_mask = np.random.randint(0, 4, ema_inputs.shape[0])


    for idx in range(ema_inputs.shape[0]):
        img = np.transpose(ema_inputs[idx].cpu().numpy(), (1, 2, 0))
        new_img1 = np.expand_dims(np.pad(img[:, :, 0], (scale_mask[idx], scale_mask[idx]), 'edge'), axis=-1)
        new_img2 = np.expand_dims(np.pad(img[:, :, 1], (scale_mask[idx], scale_mask[idx]), 'edge'), axis=-1)
        new_img3 = np.expand_dims(np.pad(img[:, :, 2], (scale_mask[idx], scale_mask[idx]), 'edge'), axis=-1)
        new_img = np.concatenate([new_img1, new_img2, new_img3], axis=-1)

        if shift_mask[idx] == 0:
            ema_inputs[idx] = torch.from_numpy(new_img[0:image_size, 0:image_size,:].transpose((2,0,1))).cuda()
        elif shift_mask[idx] == 1:
            ema_inputs[idx] = torch.from_numpy(new_img[-image_size:, 0:image_size,:].transpose((2,0,1))).cuda()
        elif shift_mask[idx] == 2:
            ema_inputs[idx] = torch.from_numpy(new_img[0:image_size, -image_size:,:].transpose((2,0,1))).cuda()
        elif shift_mask[idx] == 3:
            ema_inputs[idx] = torch.from_numpy(new_img[-image_size:, -image_size:,:].transpose((2,0,1))).cuda()

    return ema_inputs, shift_mask, scale_mask

def transforms_output_for_shift(ema_inputs, shift_mask, scale_mask, image_size):

    for idx in range(ema_inputs.shape[0]):
        # shift back
        img = np.transpose(ema_inputs[idx].cpu().numpy(), (1, 2, 0))
        new_img1 = np.expand_dims(np.pad(img[:, :, 0], (scale_mask[idx], scale_mask[idx]), 'edge'), axis=-1)
        new_img2 = np.expand_dims(np.pad(img[:, :, 1], (scale_mask[idx], scale_mask[idx]), 'edge'), axis=-1)
        # new_img3 = np.expand_dims(np.pad(img[:, :, 2], (scale_mask[idx], scale_mask[idx]), 'edge'), axis=-1)
        new_img = np.concatenate([new_img1, new_img2], axis=-1)

        if shift_mask[idx] == 0:
            ema_inputs[idx] = torch.from_numpy(new_img[0:image_size, 0:image_size, :].transpose((2, 0, 1))).cuda()
        elif shift_mask[idx] == 1:
            ema_inputs[idx] = torch.from_numpy(new_img[-image_size:, 0:image_size, :].transpose((2, 0, 1))).cuda()
        elif shift_mask[idx] == 2:
            ema_inputs[idx] = torch.from_numpy(new_img[0:image_size, -image_size:, :].transpose((2, 0, 1))).cuda()
        elif shift_mask[idx] == 3:
            ema_inputs[idx] = torch.from_numpy(new_img[-image_size:, -image_size:, :].transpose((2, 0, 1))).cuda()

    return ema_inputs


def crop_output_for_shift(ema_inputs, shift_mask, scale_mask):


    ema_outputs = torch.zeros((ema_inputs.shape[0], 2, ema_inputs.shape[-1], ema_inputs.shape[-1]))

    for idx in range(ema_inputs.shape[0]):
        # shift back
        new_img = np.transpose(ema_inputs[idx].cpu().numpy(), (1, 2, 0)).copy()
        if shift_mask[idx] == 0:
            ema_outputs[idx,:,0:224-scale_mask[idx],0:224-scale_mask[idx]] = torch.from_numpy(new_img[scale_mask[idx]:, scale_mask[idx]:, :].transpose((2, 0, 1)))
        elif shift_mask[idx] == 1:
            ema_outputs[idx,:,0:224-scale_mask[idx],0:224-scale_mask[idx]] = torch.from_numpy(new_img[:-scale_mask[idx], scale_mask[idx]:, :].transpose((2, 0, 1)))
        elif shift_mask[idx] == 2:
            ema_outputs[idx,:,0:224-scale_mask[idx],0:224-scale_mask[idx]] = torch.from_numpy(new_img[scale_mask[idx]:, :-scale_mask[idx], :].transpose((2, 0, 1)))
        elif shift_mask[idx] == 3:
            ema_outputs[idx,:,0:224-scale_mask[idx],0:224-scale_mask[idx]] = torch.from_numpy(new_img[:-scale_mask[idx], :-scale_mask[idx], :].transpose((2, 0, 1)))

    ema_outputs = ema_outputs.cuda()
    return ema_outputs

def crop_output_back_shift(ema_inputs, shift_mask, scale_mask,image_size):

    ema_outputs = torch.zeros((ema_inputs.shape[0], 2 , ema_inputs.shape[-1], ema_inputs.shape[-1]))

    for idx in range(ema_inputs.shape[0]):
        # shift back
        new_img = np.transpose(ema_inputs[idx].cpu().numpy(), (1, 2, 0)).copy()
        if shift_mask[idx] == 0:
            ema_outputs[idx,:,0:224-scale_mask[idx],0:224-scale_mask[idx]] = torch.from_numpy(new_img[:-scale_mask[idx], :-scale_mask[idx], :].transpose((2, 0, 1)))
        elif shift_mask[idx] == 1:
            ema_outputs[idx,:,0:224-scale_mask[idx],0:224-scale_mask[idx]] = torch.from_numpy(new_img[scale_mask[idx]:, :-scale_mask[idx], :].transpose((2, 0, 1)))
        elif shift_mask[idx] == 2:
            ema_outputs[idx,:,0:224-scale_mask[idx],0:224-scale_mask[idx]] = torch.from_numpy(new_img[:-scale_mask[idx], scale_mask[idx]:, :].transpose((2, 0, 1)))
        elif shift_mask[idx] == 3:
            ema_outputs[idx,:,0:224-scale_mask[idx],0:224-scale_mask[idx]] = torch.from_numpy(new_img[scale_mask[idx]:, scale_mask[idx]:, :].transpose((2, 0, 1)))


    return ema_outputs.cuda()


if __name__ == '__main__':
    image = cv2.imread("dog.jpeg")
    image = np.expand_dims(image[:224, :224, :], axis=0)
    image = np.repeat(image, 8, axis=0)

    image = torch.from_numpy(image.transpose((0, 3, 1, 2)))
    print(image.shape)

    # ema_inputs, rot_mask, flip_mask = transforms_for_rot(image)


    # scale
    ema_outout2, scale_mask = transforms_for_scale(image, 224)
    print ("scale_mask", scale_mask)


    ema_outout2_final, scale_mask = transforms_back_scale(ema_outout2, scale_mask, 224)

    # ema_outout2_final = transforms_back_rot(ema_outout2_final, rot_mask, flip_mask)
    ema_outout1_final = postprocess_scale(image, scale_mask, 224)
    print (np.array_equal(ema_outout1_final, image))
    # exit(0)
    # ema_outout1_final = image
    # output2, shift_mask , scale_mask = transforms_input_for_shift(image, 224)
    # output2_final = crop_output_back_shift(output2, shift_mask, scale_mask, 224)
    #
    # output1 = transforms_output_for_shift(image, shift_mask, scale_mask, 224)
    # output1_final = crop_output_for_shift(output1, shift_mask, scale_mask)
    # print (ema_inputs.shape)
    # exit(0)
    # ema_inputs2 = postprocess_shift(ema_inputs, shift_mask, scale_mask, 224)

    # ema_inputs, shift_mask, scale_mask = transforms_back_shift(ema_inputs, shift_mask, scale_mask)
    # c = np.array_equal(output2, output1)
    # print (c)

    # c = np.array_equal(ema_outout1_final, ema_outout2_final)
    # c2 = np.allclose(ema_outout1_final, ema_outout2_final, rtol=1)
    # print (c)
    # print (c2)
    # print (ema_outout2_final[0,0,170:200,170:200])
    # print (ema_outout1_final[0, 0, 170:200, 170:200])
    # print (torch.sum((ema_outout1_final-ema_outout2_final)**2))
    # exit(0)

    # ema_back = transforms_back_rot(ema_inputs)
    # print (np.array_equal(image, ema_back))
    # import matplotlib.pyplot as plt
    # ema_outout1_final = ema_outout1_final.numpy()
    # ema_outout1_final = ema_outout1_final.transpose((0, 2, 3, 1))
    # #
    # plt.figure(1)
    # plt.imshow(ema_outout1_final[0])  # 显示图片
    # ema_outout2_final = ema_outout2_final.numpy()
    # ema_outout2_final = ema_outout2_final.transpose((0, 2, 3, 1))
    # plt.figure(2)
    # plt.imshow(ema_outout2_final[0])  # 显示图片
    # plt.show()
    # exit(0)
    #
    # plt.figure(3)
    # output2 = output2.numpy()
    # output2 = output2.transpose((0, 2, 3, 1))
    # plt.imshow(output2[0])
    #
    # plt.figure(4)
    # output1 = output1.numpy()
    # output1 = output1.transpose((0, 2, 3, 1))
    # plt.imshow(output1[0])
    # plt.show()
    #
    # exit(0)
    # plt.figure(3)
    # plt.imshow(ema_inputs[2])  # 显示图片
    # plt.figure(4)
    # plt.imshow(ema_inputs[3])  # 显示图片
    # plt.figure(5)
    # plt.imshow(ema_inputs[4])  # 显示图片
    # plt.figure(6)
    # plt.imshow(ema_inputs[5])  # 显示图片
    # plt.figure(7)
    # plt.imshow(ema_inputs[6])  # 显示图片
    # plt.figure(8)
    # plt.imshow(ema_inputs[7])  # 显示图片
    # plt.axis('off')  # 不显示坐标轴
    # plt.show()
