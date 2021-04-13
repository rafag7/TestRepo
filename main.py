import cv2
import numpy as np
import pickle
import time

print("yey")
def retira_cont_interior(cont, hierq):
    """
    Este métodos retira todos os contornos interiores.
    """
    new_cont = []
    new_hierq = [[]]
    for i in range(len(cont)):
        # se o ultimo valor da hierarquia não for -1
        # significa que é um contorno interior, e não será
        # adicionado ao novo array
        if hierq[0][i][3] == -1:
            new_cont.append(cont[i])
            new_hierq[0].append(hierq[0][i])
    return np.array(new_cont)

def obter_contornos(diff_frame):
    """
    Este metodo retorna todos os contornos de uma frame
    (Todos os contornos interiores são retirados)
    """
    cont, h = cv2.findContours(diff_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return retira_cont_interior(cont, h)


def obter_novo_id():
    """
    Este metodo retorna um novo id para um objeto
    """
    if len(cont_id) > 0:
        new_id = cont_id[-1][1]+1
    else:
        new_id = 0
    return new_id


def obter_id_objeto(x, y, w, h, obj_cor):
    """
    Retorna o id de um objeto
    """
    # calculo do centro desta bounding box
    cur_center = [x+(w//2), y+(h//2)]

    # quão perto tem de estar de um centro para
    # ser considerado o mesmo objeto
    min_distance_thresh = 20

    cur_min_dist = 10000000000
    selected_id = -1

    # itera sobre todos os centros
    for idx, [last_center, id, all_centers] in enumerate(cont_id):

        dist = np.linalg.norm(np.array(cur_center) - np.array(last_center))

        if dist < cur_min_dist:
            cur_min_dist = dist
            selected_id = id
            selected_index = idx

    if cur_min_dist < min_distance_thresh:
        # atualiza as informações deste ID
        cont_id[selected_index][0] = cur_center
        cont_id[selected_index][1] = selected_id
        cont_id[selected_index][2].append([cur_center, obj_cor])

        # retorna o id identificado
        return str(selected_id)

    # Quando não encontra um centro relativamente perto
    # assume: que se trata de um novo objeto
    new_id = obter_novo_id()
    cont_id.append([cur_center, new_id, []])
    return str(new_id)


# TODO dsadsda

def desenha_movimento(frame):
    """
    Desenha o movimento de cada objeto
    """
    global start_time
    cur_time = time.time()
    delete = False

    # tempo de vida de cada ponto mais antigo
    if cur_time - start_time > 0.03:
        start_time = cur_time
        delete = True

    # itera sobre todos os objetos e desenha o seu rastro
    for idx, [last_center, id, all_centers] in enumerate(cont_id):
        for center, color in all_centers:
            cv2.circle(frame, tuple(center), radius=0, color=color, thickness=3)

        # retira o ponto mais antigo quando acaba o tempo
        if delete:
            if len(all_centers) > 0:
                all_centers.pop(0)


def classificar(diff_frame, frame, diff_factor=1600, obj_min_area=133, mostrar_cont=False):
    """
    Classifica as regiões ativas de uma imagem
    """
    cont = obter_contornos(diff_frame)

    # para ser uma pessoa a altura tem de ser [n] vezes
    # maior que a largura
    pessoa_min_ratio = 1.70

    for i in cont:
        area = cv2.contourArea(i)
        x, y, w, h = cv2.boundingRect(i)

        if area > obj_min_area:
            if(area < diff_factor):
                if(h/w < pessoa_min_ratio):
                    cv2.rectangle(frame, (x, y), (x+w, y+h), outro_box_cor, 1)
                else:
                    obj_id = obter_id_objeto(x, y, w, h, pessoa_rastro_cor)

                    cv2.rectangle(frame, (x, y), (x+w, y+h), pessoa_box_cor, 1)
                    cv2.putText(frame, obj_id, (x+(w//2), y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, pessoa_box_cor, thickness=2)
            elif(area >= diff_factor):
                obj_id = obter_id_objeto(x, y, w, h, carro_rastro_cor)
                cv2.rectangle(frame, (x, y), (x+w, y+h), carro_box_cor, 1)
                cv2.putText(frame, obj_id, (x+(w//2), y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, carro_box_cor, thickness=2)
            if mostrar_cont:
                cv2.drawContours(frame, cont, -1, (255, 255, 255), 1)
            # cv2.putText(frame, str(area), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2)

    desenha_movimento(frame)
    return frame


def processar_video(cap, bg, vid_speed=2):
    """
    Processa todas as frames de um video, atribuindo uma classificação aos seus objetos
    """
    pause = False
    bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

    # parametros
    bnw_threshold = 15
    median_blur_amount = 7
    kernel_size = 7
    mostrar_cont = False

    while(cap.isOpened()):
        if not pause:
            ret, frame = cap.read()

            # video terminou
            if frame is None:
                break

            # converte a frame para tons de cinzento
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # subtrai o fundo da frame atual
            sub = cv2.absdiff(frame_gray, bg_gray)

            # conversão para uma imagem binaria
            _, diff_frame = cv2.threshold(sub, bnw_threshold, 255, cv2.THRESH_BINARY)

            # melhoramentos da imagem
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            diff_frame = cv2.medianBlur(diff_frame, median_blur_amount)

            kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
            diff_frame = cv2.dilate(diff_frame, kernel2, iterations=3)
            diff_frame = cv2.erode(diff_frame, kernel2, iterations=4)
            diff_frame = cv2.dilate(diff_frame, kernel2, iterations=1)

            frame = classificar(diff_frame, frame, mostrar_cont=mostrar_cont)

            # Controlos e informações
            cv2.rectangle(frame, (5, len(frame)-80), (15, len(frame)-70), outro_box_cor, -1)
            cv2.putText(frame, "Outro", (20, len(frame)-71),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), thickness=1)
            cv2.rectangle(frame, (5, len(frame)-100), (15, len(frame)-90), carro_box_cor, -1)
            cv2.putText(frame, "Carro", (20, len(frame)-91),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), thickness=1)
            cv2.rectangle(frame, (5, len(frame)-120), (15, len(frame)-110), pessoa_box_cor, -1)
            cv2.putText(frame, "Pessoa", (20, len(frame)-111),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), thickness=1)
            cv2.putText(frame, "frame: "+str(int(cap.get(cv2.CAP_PROP_POS_FRAMES))), (5, len(frame)-55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), thickness=1)
            cv2.putText(frame, "[p] - pausar/continuar", (5, len(frame)-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), thickness=1)
            cv2.putText(frame, "[x] - terminar", (5, len(frame)-25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), thickness=1)
            cv2.putText(frame, "[c] - regioes ativas (" + str(mostrar_cont)+")", (5, len(frame)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), thickness=1)

        cv2.imshow('Regioes Ativas', diff_frame)
        cv2.imshow('Frame', frame)

        key = cv2.waitKey(vid_speed)

        if key & 0xFF == ord('c'):
            mostrar_cont = not mostrar_cont
        if key & 0xFF == ord('p'):
            pause = not pause
        elif key & 0xFF == ord('x'):
            break


if __name__ == "__main__":
    capture = cv2.VideoCapture('camera1.mov')
    bg_image = cv2.imread("bg_img.png")
    cont_id = []
    start_time = time.time()

    # cores
    outro_box_cor = (0, 255, 0)
    pessoa_box_cor = (0, 0, 255)
    carro_box_cor = (255, 0, 0)
    pessoa_rastro_cor = (255, 255, 0, 50)
    carro_rastro_cor = (0, 255, 255, 50)

    processar_video(capture, bg_image)

    # print("\nN Objetos", len(cont_id))
