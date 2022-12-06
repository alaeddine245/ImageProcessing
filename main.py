import numpy as np
import random
import matplotlib.pyplot as plt
import copy
import cv2
class pnm:
    def __init__(self, format_, comment, lx,ly,max_pixel, image_mat):
        self.format_= format_
        self.comment = comment
        self.lx = lx
        self.ly = ly
        self.max_pixel = max_pixel
        self.image_mat = image_mat
    def get_values(self):
        return self.format_, self.comment, self.lx,self.ly, self.max_pixel, self.image_mat    
class Pgm(pnm):
    def show(self):
        plt.imshow(self.image_mat, cmap='gray')
        plt.show()
    pass
class Ppm(pnm):
    def show(self):
        plt.imshow(self.image_mat)
        plt.show()
    pass
class PgmOperations:
    def read(self, path):
        image = open(path)
        format_ = image.readline().strip()
        comment = image.readline().strip()
        lx, ly = [int(c) for c in image.readline().split()]
        max_pixel = int(image.readline())
        image_mat=[]
        for line in image.readlines():
            image_mat.append([int(c) for c in line.strip().split(' ')])
        image_mat = np.array(image_mat)
        image.close()
        return Pgm(format_, comment, lx,ly,max_pixel, image_mat)
    def write(self, pgm, path):
        image= open(path, "w")
        image.write(pgm.format_+ '\n')
        image.write(pgm.comment + '\n')
        
        image.write(str(pgm.lx)+ ' '+ str(pgm.ly) + '\n')
        image.write(str(pgm.max_pixel) + '\n')

        np.savetxt(image, pgm.image_mat, fmt = '%d', delimiter=' ', header= '', comments='')
        image.close()
    
    def moyenne(self, pgm):
        return np.sum(pgm.image_mat) / (pgm.lx * pgm.ly)
    def ecart_type(self, pgm):
        return np.sqrt(np.sum(np.power((pgm.image_mat-self.moyenne(pgm)), 2))/(pgm.lx * pgm.ly))
    def histogram(self, pgm):
        histogram=[]
        for i in range(pgm.max_pixel+1):
            histogram.append(0)
        for arr in pgm.image_mat:
            for el in arr:
                histogram[el]+=1
        return np.array(histogram)
    def histogram_cumul(self, pgm):
        histogram = self.histogram(pgm)
        for i in range(1,pgm.max_pixel+1):
            histogram[i]=histogram[i]+histogram[i-1]
        return np.array(histogram)
    def histogram_egalise(self, pgm):
        histogram_cum = self.histogram_cumul(pgm)
        histogram = self.histogram(pgm)
        proba = histogram_cum / (pgm.lx*pgm.ly)
        n1 = np.floor(255*proba).astype(int)
        j=0
        result = np.zeros((256,))
        for el in n1:
            result[el]+=histogram[j]
            j+=1
        return result
    def draw_histogram(self, histogram):
        plt.plot(range(256), histogram)
        plt.xlabel('grey scale')
        plt.show()
    def noise(self, pgm):
        pgm_copy = copy.deepcopy(pgm)
        rand_number  = random.randint(0,20)
        for i in range(pgm_copy.ly):
            for j in range(pgm_copy.lx):
                rand_number  = random.randint(0,20)
                if rand_number ==0:
                    pgm_copy.image_mat[i][j] = 0
                if rand_number == 20:
                    pgm_copy.image_mat[i][j] =255
        return pgm_copy
    def mean_filter(self, pgm, n):
        pgm_copy = copy.deepcopy(pgm)
        kernel = np.empty(shape=(n,n))
        kernel.fill(1/n)
        new_mat=np.zeros(shape=pgm_copy.image_mat.shape)
        for i in range(pgm_copy.ly-n+1):
            for j in range(pgm_copy.lx-n+1):
                new_mat[i,j] = np.sum(pgm_copy.image_mat[i:i+n, j:j+n]*kernel)
        pgm_copy.image_mat = new_mat
        return pgm_copy
    def median_filter(self, pgm, n):
        pgm_copy = copy.deepcopy(pgm)
        new_mat=np.zeros(shape=pgm_copy.image_mat.shape)
        for i in range(pgm_copy.ly-n+1):
            for j in range(pgm_copy.lx-n+1):
                new_mat[i,j] = np.median(pgm_copy.image_mat[i:i+n, j:j+n])
        pgm_copy.image_mat = new_mat
        return pgm_copy
    def highpassing_filter(self, pgm):
        pgm_copy = copy.deepcopy(pgm)
        n=5
        kernel = np.array([[-1, -1, -1, -1, -1],
                   [-1,  1,  2,  1, -1],
                   [-1,  2,  4,  2, -1],
                   [-1,  1,  2,  1, -1],
                   [-1, -1, -1, -1, -1]])
        new_mat=np.zeros(shape=pgm_copy.image_mat.shape)
        for i in range(pgm_copy.ly-n+1):
            for j in range(pgm_copy.lx-n+1):
                new_mat[i,j] = np.sum(pgm_copy.image_mat[i:i+n, j:j+n]*kernel)
        pgm_copy.image_mat = new_mat
        return pgm_copy
    
    def get_SNB(self,pgm,fil_pgm):
        return np.sqrt(np.sum(np.power(pgm.image_mat- np.mean(pgm.image_mat), 2))/np.sum(np.power(fil_pgm.image_mat - pgm.image_mat, 2)))

class PpmOperations:
    def read(self, path):
        image = open(path, 'rb')
        format_ = image.readline().strip()
        comment = image.readline().strip()
        lx, ly = [int(c) for c in image.readline().split()]
        max_pixel = int(image.readline().strip())
        byte_list = image.read()
        image_mat =  np.flip(np.array(list(byte_list)).reshape(-1,3), axis=1)
        image.close()
        return Ppm(format_, comment, lx,ly,max_pixel, image_mat)
    def threshhold(self, image, thresh, cond):
        R,G,B = thresh
        return image
    def show(self, ppm):
        plt.imshow(ppm)
        plt.show()


"""   
pgm_ops = PgmOperations()
ppm_ops = PpmOperations()
pgm = pgm_ops.read("chat.pgm")
ppm = ppm_ops.read('chat.ppm')
print(ppm)
new_ppm = ppm_ops.threshhold(ppm, (100,100,100), 'OR')
ppm_ops.show(new_ppm.astype(float))




print(pgm.get_pgm_values())
print(pgm_ops.moyenne(pgm))
print(pgm_ops.ecart_type(pgm))
print(pgm_ops.histogram(pgm))
print(pgm_ops.histogram_cumul(pgm))
print(pgm_ops.histogram_egalise(pgm))
print(pgm_ops.show(pgm_ops.noise(pgm)))
print(pgm_ops.show(pgm_ops.mean_filter(pgm_ops.noise(pgm), n=3)))
print(pgm_ops.get_SNB(pgm, pgm_ops.noise(pgm)))
print(pgm_ops.get_SNB(pgm, pgm_ops.median_filter(pgm_ops.noise(pgm), n=3)))
print(pgm_ops.get_SNB(pgm, pgm_ops.mean_filter(pgm_ops.noise(pgm), n=3)))
print(pgm_ops.show(pgm_ops.median_filter(pgm_ops.noise(pgm), n=3)))
"""
#pgm_ops.write(pgm, "new.pgm")
