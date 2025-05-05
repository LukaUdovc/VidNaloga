import cv2 as cv
import numpy as np
import random


def manhattan(a, b):
    return np.sum(np.abs(a - b))

def kmeans(slika, k=3, iteracije=10,dimenzija=3):
    '''Izvede segmentacijo slike z uporabo metode k-means z Manhattan razdaljo.'''
    h, w, _ = slika.shape
    podatki = []

    # Pripravi podatke
    for y in range(h):
        for x in range(w):
                b, g, r = slika[y, x]
                if dimenzija == 3:
                    podatki.append(np.array([b, g, r]))
                else:
                    podatki.append(np.array([b, g, r, x, y]))

    podatki = np.array(podatki)
    
    # Inicializacija centrov (naključno)
    centri = podatki[np.random.choice(len(podatki), k, replace=False)]

    for _ in range(iteracije):
        oznake = []
        for p in podatki:
            razdalje = [manhattan(p, c) for c in centri]
            oznake.append(np.argmin(razdalje))
        oznake = np.array(oznake)

        # Posodobitev centrov
        novi_centri = []
        for i in range(k):
            skupina = podatki[oznake == i]
            if len(skupina) > 0:
                novi_centri.append(np.mean(skupina, axis=0))
            else:
                novi_centri.append(centri[i])
        centri = np.array(novi_centri)

    # Rekonstrukcija slike
    rezultat = np.zeros((h * w, 3), dtype=np.uint8)
    for i, oznaka in enumerate(oznake):
        rezultat[i] = centri[oznaka][:3]
    rezultat = rezultat.reshape((h, w, 3))
    return rezultat

def meanshift(slika, h=30, dimenzija=5, max_iter=10, min_cd=20):
    '''Izvede segmentacijo slike z uporabo metode mean-shift (barvni prostor BGR).'''
    h_sl, w_sl, _ = slika.shape
    podatki = []

    for y in range(h_sl):
        for x in range(w_sl):
            b, g, r = slika[y, x]
            if dimenzija == 3:
                podatki.append(np.array([b, g, r]))
            else:
                podatki.append(np.array([b, g, r, x, y]))

    podatki = np.array(podatki)
    konvergirani = []

    # Premik posamezne točke
    for xi in podatki:
        tocka = xi.copy()
        for _ in range(max_iter):
            razdalje = np.linalg.norm(podatki - tocka, axis=1)
            utezi = np.exp(- (razdalje ** 2) / (2 * h ** 2))
            nova_tocka = np.sum(podatki * utezi[:, np.newaxis], axis=0) / np.sum(utezi)
            if np.linalg.norm(nova_tocka - tocka) < 1:
                break
            tocka = nova_tocka
        konvergirani.append(tocka)

    # Združevanje konvergiranih točk v centre
    centri = []
    oznake = []
    for p in konvergirani:
        dodano = False
        for i, c in enumerate(centri):
            if np.linalg.norm(p - c) < min_cd:
                oznake.append(i)
                dodano = True
                break
        if not dodano:
            centri.append(p)
            oznake.append(len(centri) - 1)

    # Rekonstrukcija slike
    centri = np.array(centri)
    rezultat = np.zeros((h_sl * w_sl, 3), dtype=np.uint8)
    for i, oznaka in enumerate(oznake):
        rezultat[i] = centri[oznaka][:3]
    return rezultat.reshape((h_sl, w_sl, 3))

def izracunaj_centre(slika, izbira='nakljucno', dimenzija_centra=3, T=30):
    '''Izračuna centre za metodo k-means.'''
    h, w, _ = slika.shape
    podatki = []

    for y in range(h):
        for x in range(w):
            b, g, r = slika[y, x]
            if dimenzija_centra == 3:
                podatki.append(np.array([r, g, b]))
            else:
                podatki.append(np.array([r, g, b, x, y]))

    podatki = np.array(podatki)
    centri = []

    if izbira == 'nakljucno':
        while len(centri) < 3:
            kandidat = podatki[random.randint(0, len(podatki) - 1)]
            if all(np.linalg.norm(c - kandidat) > T for c in centri):
                centri.append(kandidat)

    return np.array(centri)

if __name__ == "__main__":
    slika = cv.imread("zelenjava.jpg")
    slika = cv.resize(slika, (slika.shape[1] // 4, slika.shape[0] // 4))
    seg_barva = kmeans(slika, k=6, iteracije=10, dimenzija=3)
    seg_barva_lokacija = kmeans(slika, k=6, iteracije=10, dimenzija=5)
    primerjava = np.hstack((seg_barva, seg_barva_lokacija))
    cv.imshow("Levo: RGB     Desno: RGB + XY", primerjava)
    cv.imwrite("primer_rgb_vs_rgb_xy.png", primerjava)
    cv.waitKey(0)
    slika = cv.resize(slika, (slika.shape[1] // 2, slika.shape[0] // 2))

    print("Mean-shift: barva (3), h=20, min_cd=25")
    seg_m1 = meanshift(slika, h=20, dimenzija=3, min_cd=25)
    cv.imshow("Mean-shift (3): h=20", seg_m1)
    
    cv.waitKey(0)
    cv.destroyAllWindows()