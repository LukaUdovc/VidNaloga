import cv2 as cv
import numpy as np

def kmeans(slika, k=3, iteracije=10):
    Z = slika.reshape((-1, 3)).astype(np.float32)
    kriterij = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, iteracije, 1.0)
    _, labeli, centri = cv.kmeans(Z, k, None, kriterij, 10, cv.KMEANS_RANDOM_CENTERS)
    centri = np.uint8(centri)
    rezultat = centri[labeli.flatten()].reshape(slika.shape)
    return rezultat

def meanshift(slika, prostorska_rad=10, barvna_rad=10):
    filtrirana = cv.pyrMeanShiftFiltering(slika, prostorska_rad, barvna_rad)
    return filtrirana

if __name__ == "__main__":
    slika = cv.imread("slike/zelenjava.jpg")

    # Segmentacija z K-means
    rezultat_kmeans = kmeans(slika)
    cv.imwrite("out_kmeans.jpg", rezultat_kmeans)

    # Segmentacija z Mean-Shift
    rezultat_meanshift = meanshift(slika)
    cv.imwrite("out_meanshift.jpg", rezultat_meanshift)

    print("Shranjeni sta: out_kmeans.jpg in out_meanshift.jpg")
