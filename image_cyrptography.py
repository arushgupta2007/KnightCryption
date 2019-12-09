import argparse, cv2, numpy as np, os
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Image Cryptography Using Knight's Tour")
parser.add_argument("--path", type=str, help="Path to Image")

args = parser.parse_args()
knight_tour_8 = [
    [34, 51, 30, 7, 38, 53, 26, 11],
    [31, 6, 33, 52, 27, 10, 39, 54],
    [50, 35, 8, 29, 56, 37, 12, 25],
    [5, 32, 49, 36, 9, 28, 55, 40],
    [48, 63, 20, 3, 44, 57, 24, 13],
    [19, 4, 45, 64, 21, 16, 41, 58],
    [62, 47, 2, 17, 60, 43, 14, 23],
    [1, 18, 61, 46, 15, 22, 59, 42]
]

knight_tour = [
    [1,	126, 191, 196, 5, 122, 187, 200, 9, 120, 183, 202, 11, 118,	181, 204],
    [192,	195,	2,	125,	188,	199,	6,	121,	184,	201,	10,	119,	182,	203,	12,	117],
    [127,	190,	193,	4,	197,	186,	123,	8,	211,	18,	111,	176,	113,	180,	205,	14],
    [194,	3,	128,	189,	124,	7,	198,	185,	110,	175,	210,	17,	208,	13,	116,	179],
    [129,	254,	63,	68,	171,	252,	21,	70,	173,	212,	19,	112,	177,	114,	15,	206],
    [64,	67,	130,	253,	22,	69,	172,	251,	20,	109,	174,	209,	16,	207,	178,	115],
    [255,	132,	65,	62,	249,	170,	71,	24,	213,	168,	107,	26,	215,	166,	105,	28],
    [66,	61,	256,	131,	72,	23,	250,	169,	108,	25,	214,	167,	106,	27,	216,	165],
    [133,	244,	47,	76,	147,	248,	45,	88,	149,	232,	43,	92,	151,	228,	29,	104],
    [60,	77,	146,	245,	46,	73,	148,	233,	44,	89,	150,	229,	42,	93,	164,	217],
    [243,	134,	75,	48,	247,	144,	87,	50,	231,	154,	91,	40,	227,	152,	103,	30],
    [78,	59,	246,	145,	74,	49,	234,	143,	90,	39,	230,	153,	94,	41,	218,	163],
    [135,	242,	55,	82,	139,	238,	51,	86,	155,	224,	35,	96,	159,	226,	31,	102],
    [58,	79,	138,	239,	54,	83,	142,	235,	38,	97,	158,	225,	34,	95,	162,	219],
    [241,	136,	81,	56,	237,	140,	85,	52,	223,	156,	99,	36,	221,	160,	101,	32],
    [80,	57,	240,	137,	84,	53,	236,	141,	98,	37,	222,	157,	100,	33,	220,	161]
]

path = args.path
img = cv2.imread(r"C:\\Users\\Alok\\Desktop\\gslv.png")
img = cv2.resize(img, (1600, 1600))
h, w, _ = img.shape

class Encrypt:
    def __init__(self, img, com1, com2, path_save):
        self.img = img
        self.com1 = com1
        self.com2 = com2
        self.path_save = path_save
        self.list_big_square_coor = self.list_of_squares_func(self.img, self.com1)

    def list_of_squares_func(self, img, square):
        list_ = []
        for i in range(square):
            list_.append([(0, int(h * i / square)), (int(w / square), int(h * (i + 1) / square))])

            for b in range(square-1):
                to_append = []
                x1 = int(w * (b + 1) / square)
                y1 = int(h * i / square)
                x2 = int(w * (b + 2) / square)
                y2 = int(h * (i + 1) / square)
                to_append.append((x1, y1))
                to_append.append((x2, y2))
                list_.append(to_append)

        return list_

    def save_images(self):
        counter = 1
        for i in self.list_big_square_coor:
            x1 = i[0][0]
            y1 = i[0][1]
            x2 = i[1][0]
            y2 = i[1][1]
            roi = img[y1:y2, x1:x2] 
            cv2.imwrite(self.path_save + str(counter) + ".png", roi)
            counter+=1

    def make_image(self, img_parent, list_of_sq, x):
        list_of_rows_in_func = []
        for b in range(x):
            row = []
            for i in range(x):
                if x == 8:
                    index = knight_tour_8[b][i] - 1
                else:
                    index = knight_tour[b][i] - 1
                img_read = list_of_sq[index]
                roi = img_parent[img_read[0][1]:img_read[1][1], img_read[0][0]:img_read[1][0]]
                row.append(roi)
                
            row_image = np.concatenate((row[0], row[1]), axis=1)
            for i in range(x-2):
                row_image = np.concatenate((row_image, row[i+2]), axis=1)
            list_of_rows_in_func.append(row_image)

        img_new = np.concatenate((list_of_rows_in_func[0], list_of_rows_in_func[1]), axis=0)

        for i in range(x-2):
            img_new = np.concatenate((img_new, list_of_rows_in_func[i+2]), axis=0)
        img_new = img_new.astype(np.uint8)
        return img_new

    def main(self):
        img_ = self.make_image(self.img, self.list_big_square_coor, self.com1)
        square_list = []
        counter = 1
        for i in self.list_big_square_coor:
            imaga = img_[i[0][1]:i[1][1], i[0][0]:i[1][0]]
            imaga = cv2.resize(imaga, (1600, 1600))
            imaga = self.make_image(imaga, self.list_of_squares_func(imaga, self.com2), self.com2)
            imaga = cv2.resize(imaga, (100, 100))
            cv2.imwrite(self.path_save + str(counter) + ".png", imaga)
            square_list.append(imaga)
            counter += 1
        row_list = []
        for c in range(self.com2):
            row = []
            for i in range(self.com2):
                # print(c * self.com2 + i)
                row.append(square_list[c * self.com2 + i])
            row_list.append(row)
        row_list_image = []
        for z in row_list:
            row_image = np.concatenate((z[0], z[1]), axis=1)
            for i in range(self.com2 - 2):
                row_image = np.concatenate((row_image, z[i+2]), axis=1)
            row_list_image.append(row_image)

        img_ = np.concatenate((row_list_image[0], row_list_image[1]), axis=0)
        for i in range(self.com2 - 2):
            img_ = np.concatenate((img_, row_list_image[i + 2]), axis=0)

        # for x in range(w):
        #     for y in range(h):
        #         channels_xy = img_[y,x]
        #         x1 = channels_xy[0] - 123
        #         x2 = channels_xy[1] - 123
        #         x3 = channels_xy[2] - 123
        #         # if x1 > 0: x1 = 0
        #         # if x2 > 0: x2 = 0
        #         # if x3 > 0: x3 = 0
        #         # print(x1)
        #         channels_xy = [x1, x2, x3]
        #         img_[y, x] = channels_xy

        print(img_.shape)
        # img_ = cv2.bitwise_not(img_)
        cv2.imwrite(self.path_save + "FINAL_ENCRYPTION.png", img_)

        return img_

class Decrypt:
    def __init__(self, img, com1, com2, path_to_save):
        self.img = img
        self.com1 = com1
        self.com2 = com2
        self.path = path_to_save
        self.list_big_square_coor = self.list_of_squares_func(self.img, com1)

    def list_of_squares_func(self, img, sq):
        list_ = []
        for i in range(sq):
            list_.append([(0, int(h * i / sq)), (int(w / sq), int(h * (i + 1) / sq))])

            for b in range(sq-1):
                to_append = []
                x1 = int(w * (b + 1) / sq)
                y1 = int(h * i / sq)
                x2 = int(w * (b + 2) / sq)
                y2 = int(h * (i + 1) / sq)
                to_append.append((x1, y1))
                to_append.append((x2, y2))
                list_.append(to_append)

        return list_

    def arrange_squares(self, img, com):
        list_of_sq = self.list_of_squares_func(img, com)
        # print(com)
        if com == 8:
            print("BlaBlaBla")
            array_knight_tour = np.array(knight_tour_8)
        else:
            # print("skfdjsdfkmfvkadfklkgmbk")
            array_knight_tour = np.array(knight_tour)
        square_list = []
        for i in range(com * com):
            result = np.where(array_knight_tour == i + 1)
            row_no = result[0][0]
            column = result[1][0]
            index = row_no * com + column
            square_ = list_of_sq[index]
            # print(img.shape)
            roi = img[square_[0][1]:square_[1][1], square_[0][0]:square_[1][0]]
            # print(roi.shape)
            square_list.append(roi)

        row_list = []
        for c in range(com):
            row = []
            for i in range(com):
                # print(c * self.com2 + i)
                row.append(square_list[c * com + i])
            row_list.append(row)
        row_list_image = []
        for z in row_list:
            row_image = np.concatenate((z[0], z[1]), axis=1)
            for i in range(com - 2):
                row_image = np.concatenate((row_image, z[i+2]), axis=1)
            row_list_image.append(row_image)

        img_ = np.concatenate((row_list_image[0], row_list_image[1]), axis=0)
        for i in range(com - 2):
            img_ = np.concatenate((img_, row_list_image[i + 2]), axis=0)

        return img_

    def main(self):
        square_list = []
        for i in self.list_big_square_coor:
            roi = self.img[i[0][1]:i[1][1], i[0][0]:i[1][0]]
            imaga = self.arrange_squares(cv2.resize(roi, (1600, 1600)), self.com2)
            imaga = cv2.resize(imaga, (100, 100))
            cv2.imwrite(r"C:\\Users\\Alok\\Desktop\\Bunch_of_64_images\\\decrypt\\" + str(self.list_big_square_coor.index(i)) + ".png", imaga)
            square_list.append(imaga)

        row_list = []
        for c in range(self.com2):
            row = []
            for d in range(self.com2):
                # print(c * self.com2 + i)
                row.append(square_list[c * self.com2 + d])
                # cv2.imwrite(r"C:\\Users\\Alok\\Desktop\\Bunch_of_64_images\\\decrypt\\" + str(c * self.com2 + d) + ".png", square_list[c * self.com2 + d])
            row_list.append(row)
        row_list_image = []
        for z in row_list:
            row_image = np.concatenate((z[0], z[1]), axis=1)
            for i in range(self.com2 - 2):
                row_image = np.concatenate((row_image, z[i+2]), axis=1)
            row_list_image.append(row_image)

        img_ = np.concatenate((row_list_image[0], row_list_image[1]), axis=0)
        for i in range(self.com2 - 2):
            img_ = np.concatenate((img_, row_list_image[i + 2]), axis=0)

        img_ = self.arrange_squares(img_, self.com1)
        img_ = cv2.GaussianBlur(cv2.medianBlur(img_, 5), (5, 5), 0)
        # for x in range(w):
        #     for y in range(h):
        #         channels_xy = img_[y,x]
        #         x1 = channels_xy[0] + 123
        #         x2 = channels_xy[1] + 123
        #         x3 = channels_xy[2] + 123
        #         # if x1 > 0: x1 = 0
        #         # if x2 > 0: x2 = 0
        #         # if x3 > 0: x3 = 0
        #         # print(x1)
        #         channels_xy = [x1, x2, x3]
        #         img_[y, x] = channels_xy
        print(img.shape)
        # img_ = cv2.bitwise_not(img_)
        cv2.imwrite(self.path + r"FINAL_DE.png", img_)
        
        return img_



if __name__ == "__main__":
    eclass_instance = Encrypt(img, 16, 16, "Bunch_of_64_images/")
    img_new = eclass_instance.main()
    dclass_instance = Decrypt(img_new, 16, 16, r"C:\\Users\\Alok\\Desktop\\Bunch_of_64_images\\\decrypt\\")
    img_d = dclass_instance.main()
    cv2.imshow("Decrypt", img_d)
    cv2.imshow("Encrypt", img_new)
    cv2.imwrite(r"C:\\Users\\Alok\\Desktop\\Encrypt_level2.png", img_new)
    cv2.imshow("Real", img)
    cv2.waitKey()
    plt.hist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).ravel(), 256, [0, 256])
    plt.hist(cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY).ravel(), 256, [0, 256])
    plt.show()
    cv2.destroyAllWindows()
