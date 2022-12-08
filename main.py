import numpy as np
import random
import matplotlib.pyplot as plt
import copy


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
        image_mat =  np.array(list(byte_list)).reshape((ly,lx, 3))
        image.close()
        return Ppm(format_, comment, lx,ly,max_pixel, image_mat)
    def threshhold(self, ppm, thresh, cond):
        image_red = np.where(ppm.image_mat[:,:,0] >= thresh[0], 255,0)
        image_green = np.where(ppm.image_mat[:,:,1] >= thresh[1], 255,0)
        image_blue = np.where(ppm.image_mat[:,:,2] >= thresh[2], 255,0)
        ppm_copy = copy.deepcopy(ppm)
        if cond=='AND':
            ppm_copy.image_mat = np.bitwise_and(image_red, image_green, image_blue)
        elif cond =='OR':
            ppm_copy.image_mat = np.bitwise_or(image_red, image_green, image_blue)
        return ppm_copy
    def histogram(self, ppm, channel):
        histogram=[]
        for i in range(ppm.max_pixel+1):
            histogram.append(0)
        for arr in ppm.image_mat[:,:,channel]:
            for el in arr:
                histogram[el]+=1
        return np.array(histogram)
    def histogram_cumule(self, ppm, channel):
        histogram = self.histogram(ppm, channel)
        for i in range(1,ppm.max_pixel+1):
            histogram[i]=histogram[i]+histogram[i-1]
        return np.array(histogram)
    def histogram_egalise(self, ppm, channel):
        histogram = self.histogram(ppm, channel)
        histogram_cum = self.histogram_cumule(ppm, channel)
        proba = histogram_cum / (ppm.lx*ppm.ly)
        n1 = np.floor(255*proba).astype(int)
        j=0
        result = np.zeros((256,))
        for el in n1:
            result[el]+=histogram[j]
            j+=1
        return result
    
    def draw_histogram(self, ppm):
        histogram_red = self.histogram(ppm, 0)
        histogram_green = self.histogram(ppm, 1)
        histogram_blue = self.histogram(ppm, 2)
        plt.title('Histogramme')
        plt.plot(range(256), histogram_red, 'r')
        plt.plot(range(256), histogram_green, 'g')
        plt.plot(range(256), histogram_blue, 'b')
        plt.show()
    def draw_histogram_cumule(self, ppm):
        histogram_red = self.histogram_cumule(ppm, 0)
        histogram_green = self.histogram_cumule(ppm, 1)
        histogram_blue = self.histogram_cumule(ppm, 2)
        plt.title('Histogramme cumulé')
        plt.plot(range(256), histogram_red, 'r')
        plt.plot(range(256), histogram_green, 'g')
        plt.plot(range(256), histogram_blue, 'b')
        plt.show()
    def draw_histogram_egalise(self, ppm):
        histogram_red = self.histogram_egalise(ppm,0)
        histogram_green = self.histogram_egalise(ppm,1)
        histogram_blue = self.histogram_egalise(ppm,2)
        plt.title('Histogramme egalisé')
        plt.plot(range(256), histogram_red, 'r')
        plt.plot(range(256), histogram_green, 'g')
        plt.plot(range(256), histogram_blue, 'b')
        plt.show()
    def class_average(self, cl, start, end):
        niv = np.arange(start, end)
        return np.sum(cl * niv) / np.sum(cl)
    def get_variance(self,hist, s):
        c0 = hist[:s]
        c1 = hist[s:]
        pc0 = np.sum(c0) / np.sum(hist)
        pc1 = np.sum(c1) / np.sum(hist)
        m = self.class_average(hist, 0, 256)
        m0 = self.class_average(c0, 0, s)
        m1 = self.class_average(c1, s, 256)
        return pc0 * (m0 - m)**2 + pc1 * (m1 - m)**2
    def otsu_thresholding(self,hist):
        max_variance = 0
        seuil = 0
        for s in range(1, 254):
            variance = self.get_variance(hist, s)
            if variance > max_variance:
                max_variance = variance
                seuil = s
        return seuil
    def erosion(self,ppm, level):
        ppm_copy = copy.deepcopy(ppm)
        image = ppm.image_mat
        image_h, image_w = image.shape
        output = np.zeros_like(image)
        for y in range(image_h - level + 1):
            for x in range(image_w - level + 1):
                eroded_pixel = np.min(image[y:y + level, x:x + level])
                output[y, x] = eroded_pixel
        ppm_copy.image_mat = output
        return ppm_copy
    def dilatation(self,ppm, level):
        ppm_copy = copy.deepcopy(ppm)
        image = ppm.image_mat
        image_h, image_w = image.shape
        output = np.zeros_like(image)
        for y in range(image_h - level + 1):
            for x in range(image_w - level + 1):
                dilated_pixel = np.max(image[y:y + level, x:x + level])
                output[y, x] = dilated_pixel
        ppm_copy.image_mat = output
        return ppm_copy
    def ouverture(self, ppm, level):
        return self.dilatation(self.erosion(ppm, level), level)
    def fermeture(self, ppm, level):
        return self.erosion(self.dilatation(ppm, level), level)
    def show(self, ppm):
        plt.imshow(ppm)
        plt.show()