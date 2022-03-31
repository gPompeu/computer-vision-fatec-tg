# Code adapted from: http://computervision.zone/courses/hand-tracking
# Adapted by: Group Fatequino
# Adapted by: Plinio Guimarães (grupo Fatequino) 2021-2o Semestre noite.
# Another implementation of HandGestures using the mediapipe package with TensorFlow

# Remarks:
# Cada dedo possui 4 pontos de marcação (nós) e a mão tem 1 ponto na base totalizando 21 IDs
# Partindo da base de cada dedo para as pontas temos os seguintes IDs:
# polegar: 1, 2, 3, 4
# indicador: 5, 6, 7, 8
# médio: 9, 10, 11, 12
# anelar: 13, 14, 15, 16
# mindinho: 17, 18, 19, 20

import cv2
import mediapipe as mp
import time
import numpy as np
import math

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(False, 1, 0.3, 0.3)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

polegar=1
indicador=2
medio=3
anelar=4
mindinho=5

#Geometria analítica
def vetor(p1, p2):
    v=np.array([p2[0]-p1[0], p2[1]-p1[1]])
    return v

def norma(v):
    n = abs(math.sqrt(v[0]**2 + v[1]**2))
    return n

def prodInt(v1, v2):
    p = v1[0]*v2[1]+v1[1]*v2[0]
    return p

def cosVet(v1, v2):
    c = prodInt(v1, v2) / (norma(v1) * norma(v2))
    return c

def angVet(v1, v2):
    a = np.arccos(cosVet(v1,v2))
    return math.degrees(math.pi-a)



def isEsticado(listaDedos, numDedo):
    #formula para identificar os IDs do dedo informado: round(ID / 4 +0.35)
    if listaDedos == []:
        return False
    B = []
    for dedo in listaDedos:
        if round(dedo[0] /4 +0.35) == numDedo:
            #TODO: usar coordenadas do dedo para verificar se está esticado
            print(dedo)
            B.append(dedo)
            
    C = np.array(B)[0:4,1:4]
    print(C)
    #precisa ajustar esses cálculos e referências. Está calculando angulos estranhos
    v1 = vetor(C[0],C[1])
    print(v1)
    v2 = vetor(C[1],C[3])
    print(v2)
    v3 = vetor(C[0],C[3])
    print(v3)
    v4 = vetor(([0,0]), C[3])
    
    print('Angulo 1: ',angVet(v1,v2))
    print('Angulo 2: ',angVet(v2,v3))
    print('Angulo 3: ',angVet(v3,v4))
    
    return True


def findPosition(imgRGB, fingerNo=-1, handNo=0, draw=True):
    lmList = []
    
    if results.multi_hand_landmarks:
        myHand = results.multi_hand_landmarks[handNo]
        for id, lm in enumerate(myHand.landmark):
            #print(id, lm)
            if (fingerNo < 0) | (fingerNo==round(id /4 +0.35)):
                h, w, c = img.shape
                #coordenadas na tela
                #obs: lm.z assume valores de -1 a 1 sendo -1 mais próximo da câmera e 1 mais afastado
                #para simplificar não foi usado o lm.z
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(cx, cy)
                # acrescenta na lista o id e coordenadas de cada dedo
                lmList.append([id, cx, cy])

                if draw==True:
                    cv2.circle(imgRGB, (cx, cy), 50, (255, id*10 ,255-(id*10)), cv2.FILLED)
                
        if draw == True:
            mpDraw.draw_landmarks(imgRGB, myHand, mpHands.HAND_CONNECTIONS)
            
    return lmList


while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = hands.process(img)
    
    #
    #print(results.multi_hand_landmarks)
    # analisa a imagem e retorna lista com posição de cada dedo ou de um em particular 
    listaDedos = findPosition(imgRGB, indicador)
    if listaDedos != []:
        print(listaDedos)
        isEsticado(listaDedos, indicador)  

    imgRGB = cv2.resize(imgRGB,(320,250),interpolation= cv2.INTER_CUBIC)

    # em processadores mais rápidos pode-se exibir o FPS. No Raspberry Pi 3 ficou constante em 1 FPS 
    #cTime = time.time()
    #fps = 1/(cTime-pTime)
    #pTime = cTime
    #cv2.putText(imgRGB, 'FPS:'+str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    
    cv2.imshow("Image", imgRGB)
    key = cv2.waitKey(1)
    
    if key==27:
        break
    
cap.release()
cv2.destroyAllWindows()
print("fim do programa")