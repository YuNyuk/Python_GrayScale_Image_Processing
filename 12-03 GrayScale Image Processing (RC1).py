import math
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import *
from tkinter.simpledialog import *
import os.path
from math import cos, sin


## 함수부
# **************
# 공통 함수부
# **************
def malloc2D(h, w, initValue=0):
    memory = [[initValue for _ in range(w)] for _ in range(h)]
    return memory


def openImage():
    global window, canvas, paper, fullname
    global inImage, outImage, inH, inW, outH, outW
    fullname = askopenfilename(parent=window, filetypes=(('RAW파일', '*.raw'), ('모든파일', '*.*')))
    # 중요! 입력 이미지 크기를 결정
    fsize = os.path.getsize(fullname)  # 파일 크기(Byte)
    inH = inW = int(math.sqrt(fsize))
    # 메모리 할당
    inImage = malloc2D(inH, inW)
    # 파일 --> 메모리
    rfp = open(fullname, 'rb')
    for i in range(inH):
        for k in range(inW):
            inImage[i][k] = ord(rfp.read(1))
    rfp.close()
    equalImage()


def saveImage():
    global window, canvas, paper, fullname
    global inImage, outImage, inH, inW, outH, outW

    if (outImage == None):  # 영상 처리를 한 적이 없다면..
        return
    wfp = asksaveasfile(parent=window, mode='wb', defaultextension='*.raw',
                        filetypes=(('RAW파일', '*.raw'), ('모든 파일', '*.*')))
    import struct
    for i in range(outH):
        for k in range(outW):
            wfp.write(struct.pack('B', outImage[i][k]))
    wfp.close()
    messagebox.showinfo('성공', wfp.name + '저장 완료!')


def displayImage():
    global window, canvas, paper, fullname
    global inImage, outImage, inH, inW, outH, outW
    # 기존의 이미지를 오픈한 적이 있으면, 캔버스 뜯어내기
    if (canvas != None):
        canvas.destroy()

    # 벽, 캔버스, 종이 설정
    window.geometry(str(outH) + 'x' + str(outW))  # "512x512"
    canvas = Canvas(window, height=outH, width=outW, bg='yellow')  # 칠판
    paper = PhotoImage(height=outH, width=outW)  # 종이
    canvas.create_image((outH // 2, outW // 2), image=paper, state='normal')
    # 메모리 --> 화면
    # for i in range(inH):
    #     for k in range(inW):
    #         r = g = b = inImage[i][k]
    #         paper.put('#%02x%02x%02x' % (r, g, b), (k, i))
    # 더블 버퍼링... 비슷한 기법 ( 모두 다 메모리 상에 출력 형태로 생성한 후에, 한방에 출력, 로딩 속도 월등)
    rgbString = ""  # 전체에 대한 16진수 문자열
    for i in range(outH):
        oneString = ""  # 한 줄에 대한 16진수 문자열
        for k in range(outW):
            r = g = b = outImage[i][k]
            oneString += '#%02x%02x%02x ' % (r, g, b)
        rgbString += '{' + oneString + '} '
    paper.put(rgbString)
    canvas.pack()


def updateWindowTitle(title):
    window.title("영상 처리 (RC 1) - " + title)


# **************
# 영상 처리 함수부
# **************

#### 화소 점 처리

def equalImage():  # 동일 이미지
    global window, canvas, paper, fullname
    global inImage, outImage, inH, inW, outH, outW
    # 중요! 출력 영상 크기 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW
    # 메모리 할당
    outImage = malloc2D(outH, outW)
    ### 진짜 영상 처리 알고리즘 ###
    for i in range(inH):
        for k in range(inW):
            outImage[i][k] = inImage[i][k]
    ######################
    displayImage()
    updateWindowTitle("원본 이미지")


def brightImage():
    global window, canvas, paper, fullname
    global inImage, outImage, inH, inW, outH, outW
    # 중요! 출력 영상 크기 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW
    # 메모리 할당
    outImage = malloc2D(outH, outW)
    value = askinteger('정수 입력', '-255 ~ 255 입력', maxvalue=255)
    for i in range(inH):
        for k in range(inW):
            px = inImage[i][k] + value
            if (px > 255):
                px = 255
            if (px < 0):
                px = 0
            outImage[i][k] = px
    displayImage()
    updateWindowTitle("밝기 조정")


def grayImage():  # 흑백 처리
    global window, canvas, paper, fullname
    global inImage, outImage, inH, inW, outH, outW
    # 중요! 출력 영상 크기 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW
    # 메모리 할당
    outImage = malloc2D(outH, outW)

    ### 영상처리 알고리즘 ###
    for i in range(inH):
        for k in range(inW):
            if (inImage[i][k] < 127):
                outImage[i][k] = 0
            else:
                outImage[i][k] = 255
    displayImage()
    updateWindowTitle("흑백")


def andImage():
    global window, canvas, paper, fullname
    global inImage, outImage, inH, inW, outH, outW

    # 출력 이미지의 크기 결정
    outH = inH
    outW = inW
    # 출력 이미지 메모리 확보
    outImage = malloc2D(outH, outW)
    # 정수 값 입력
    val = askinteger('정수 입력', '양수 값 입력', maxvalue=255)

    # 비트 AND 연산 수행
    for i in range(inH):
        for k in range(inW):
            if (inImage[i][k] & val) >= 255:
                outImage[i][k] = 255
            elif (inImage[i][k] & val) < 0:
                outImage[i][k] = 0
            else:
                outImage[i][k] = inImage[i][k] & val
    displayImage()
    updateWindowTitle("AND 연산")


def orImage():  # Or 처리 알고리즘
    global window, canvas, paper, fullname
    global inImage, outImage, inH, inW, outH, outW

    # 출력 이미지의 크기 결정
    outH = inH
    outW = inW
    # 출력 이미지 메모리 확보
    outImage = malloc2D(outH, outW)
    # 정수 값 입력
    val = askinteger('정수 입력', '양수 값 입력', maxvalue=255)

    for i in range(inH):
        for k in range(inW):
            if (inImage[i][k] | val) >= 255:
                outImage[i][k] = 255
            elif (inImage[i][k] | val) < 0:
                outImage[i][k] = 0
            else:
                outImage[i][k] = (inImage[i][k] | val)
    displayImage()
    updateWindowTitle("OR 연산")


def xorImage():  # XOR 처리 알고리즘
    global window, canvas, paper, fullname
    global inImage, outImage, inH, inW, outH, outW

    # 출력 이미지의 크기 결정
    outH = inH
    outW = inW

    # 출력 이미지 메모리 확보
    outImage = malloc2D(outH, outW)

    # 정수 값 입력
    val = askinteger('정수 입력', '양수 값 입력', maxvalue=255)

    # 입력 배열 --> 출력배열
    for i in range(inH):
        for k in range(inW):
            xor_result = inImage[i][k] ^ val
            if xor_result >= 255:
                outImage[i][k] = 255
            elif xor_result < 0:
                outImage[i][k] = 0
            else:
                outImage[i][k] = xor_result
    displayImage()
    updateWindowTitle("XOR 연산")


def gammaImage():  # 감마 처리 알고리즘
    global window, canvas, paper, fullname
    global inImage, outImage, inH, inW, outH, outW

    # 출력 이미지의 크기 결정
    outH = inH
    outW = inW
    # 출력 이미지 메모리 확보
    outImage = malloc2D(outH, outW)
    # 감마 값 입력
    gamma = askfloat('감마 값 입력', '0 ~ 10 입력', maxvalue=10)
    for i in range(inH):
        for k in range(inW):
            # 픽셀 값 0~1로 정규화
            normalized_pixel = inImage[i][k] / 255.0
            # 감마 함수 적용 후 보정된 값 계산
            corrected_pixel = pow(normalized_pixel, gamma) * 255.0

            # 값이 0 미만이면 0으로, 255를 초과하면 255로 제한
            corrected_pixel = max(0, min(corrected_pixel, 255))

            outImage[i][k] = int(corrected_pixel)
    displayImage()
    updateWindowTitle("감마 처리")


def parabolCapImage():  # 파라볼라 Cap 처리 알고리즘

    global window, canvas, paper, fullname
    global inImage, outImage, inH, inW, outH, outW

    # 출력 이미지의 크기 결정
    outH = inH
    outW = inW
    # 출력 이미지 메모리 확보
    outImage = malloc2D(outH, outW)

    for i in range(inH):
        for k in range(inW):
            outImage[i][k] = int(255 - 255 * pow((inImage[i][k] / 128 - 1), 2))
    displayImage()
    updateWindowTitle("파라볼라 CAP")


def parabolCupImage():  # 파라볼라 Cup 처리 알고리즘

    global window, canvas, paper, fullname
    global inImage, outImage, inH, inW, outH, outW

    # 출력 이미지의 크기 결정
    outH = inH
    outW = inW
    # 출력 이미지 메모리 확보
    outImage = malloc2D(outH, outW)

    for i in range(inH):
        for k in range(inW):
            outImage[i][k] = int(255 * pow((inImage[i][k] / 128 - 1), 2))
    displayImage()
    updateWindowTitle("파라볼라 CUP")


def reverseImage():  # 반전

    global window, canvas, paper, fullname
    global inImage, outImage, inH, inW, outH, outW

    # 출력 이미지의 크기 결정
    outH = inH
    outW = inW
    # 출력 이미지 메모리 확보
    outImage = malloc2D(outH, outW)

    for i in range(inH):
        for k in range(inW):
            outImage[i][k] = 255 - inImage[i][k]

    displayImage()
    updateWindowTitle("반전")


#### 히스토그램 처리

def histoStretch():
    global window, canvas, paper, fullname
    global inImage, outImage, inH, inW, outH, outW

    # 출력 이미지의 크기 결정
    outH = inH
    outW = inW
    # 출력 이미지 메모리 확보
    outImage = malloc2D(outH, outW)
    high = inImage[0][0]
    low = inImage[0][0]
    for i in range(inH):
        for k in range(inW):
            if inImage[i][k] < low:
                low = inImage[i][k]
            if inImage[i][k] > high:
                high = inImage[i][k]

    for i in range(inH):
        for k in range(inW):
            old = inImage[i][k]
            new = int((old - low) / (high - low) * 255.0)
            if new > 255:
                new = 255
            if new < 0:
                new = 0
            outImage[i][k] = new
    displayImage()
    updateWindowTitle("히스토그램 스트레칭")


def histoEqual():  # 히스토그램 평활화
    global window, canvas, paper, fullname
    global inImage, outImage, inH, inW, outH, outW

    # 출력 이미지의 크기 결정
    outH = inH
    outW = inW
    # 출력 이미지 메모리 확보
    outImage = malloc2D(outH, outW)

    # 1단계 : 빈도 수 세기 (히스토그램의)
    histo = [0.] * 256
    for i in range(inH):
        for k in range(inW):
            histo[inImage[i][k]] += 1

    # 2단계 : 누적 히스토그램 생성
    sumHisto = [0.] * 256
    sumHisto[0] = histo[0]
    for i in range(256):
        sumHisto[i] = sumHisto[i - 1] + histo[i]

    # 3단계 : 정규화된 히스토그램 생성 normalHisto = sumHisto * (1.0 / (inH * inW) * 255.0
    normalHisto = [1.0] * 256
    for i in range(256):
        normalHisto[i] = sumHisto[i] * (1.0 / (inH * inW)) * 255.0

    # 4단계 : inImage를 정규화된 값으로 치환
    for i in range(inH):
        for k in range(inW):
            outImage[i][k] = int(normalHisto[inImage[i][k]])
    displayImage()
    updateWindowTitle("평활화")


def endIn():  # 엔드-인
    global window, canvas, paper, fullname
    global inImage, outImage, inH, inW, outH, outW

    # 출력 이미지의 크기 결정
    outH = inH
    outW = inW
    # 출력 이미지 메모리 확보
    outImage = malloc2D(outH, outW)
    high = inImage[0][0]
    low = inImage[0][0]
    for i in range(inH):
        for k in range(inW):
            if (inImage[i][k] < low):
                low = inImage[i][k] < low
            if (inImage[i][k] > high):
                high = inImage[i][k]
    high -= 50
    low += 50

    for i in range(inH):
        for k in range(inW):
            old = inImage[i][k]
            new = (int)((float)(old - low) / (float)(high - low) * 255.0)
            if (new > 255):
                new = 255
            if (new < 0):
                new = 0
            outImage[i][k] = new
    displayImage()
    updateWindowTitle("엔드-인")


#### 기하학적 처리

def zoomIn1():  # 포워딩 확대 알고리즘
    global window, canvas, paper, fullname
    global inImage, outImage, inH, inW, outH, outW

    # 정수 값 입력
    scale = askinteger('배율 입력', '정수 값 ', maxvalue=4)

    # 출력 이미지의 크기 결정
    outH = (int)(inH * scale);
    outW = (int)(inW * scale);

    # 출력 이미지 메모리 확보
    outImage = malloc2D(outH, outW)

    for i in range(inH):
        for k in range(inW):
            outImage[(int)(i * scale)][(int)(k * scale)] = inImage[i][k]
    displayImage()
    updateWindowTitle("확대(포워딩)")


def zoomIn2():  # 백워딩 확대 알고리즘
    global window, canvas, paper, fullname
    global inImage, outImage, inH, inW, outH, outW

    # 정수 값 입력
    scale = askinteger('배율 입력', '정수 값 ', maxvalue=4)

    # 출력 이미지의 크기 결정
    outH = (int)(inH * scale);
    outW = (int)(inW * scale);

    # 출력 이미지 메모리 확보
    outImage = malloc2D(outH, outW)

    for i in range(outH):
        for k in range(outH):
            outImage[i][k] = inImage[(int)(i / scale)][(int)(k / scale)]
    displayImage()
    updateWindowTitle("확대(백워딩)")


def zoomIn3():  # 확대 (양선형 보간)
    global window, canvas, paper, fullname
    global inImage, outImage, inH, inW, outH, outW

    # 정수 값 입력
    scale = askinteger('배율 입력', '정수 값 ', maxvalue=4)

    # 출력 이미지의 크기 결정
    outH = (int)(inH * scale);
    outW = (int)(inW * scale);

    # 출력 이미지 메모리 확보
    outImage = malloc2D(outH, outW)

    rowRatio = (inH - 1) / (outH - 1)
    colRatio = (inW - 1) / (outW - 1)

    # 출력 이미지의 각 픽셀에 대해 양선형 보간법 수행
    for i in range(outH):
        for k in range(outH):
            # 현재 픽셀의 위치를 기준으로 입력 이미지에서 가장 가까운 네 개의 픽셀을 찾음
            baseRow = (int)(round(i * rowRatio))
            baseCol = (int)(round(k * colRatio))
            # 현재 픽셀의 위치와 가장 가까운 네 개의 픽셀 사이의 거리를 계산
            dx = (i * rowRatio) - baseRow
            dy = (k * colRatio) - baseCol

            if (baseRow >= 0 and baseRow < inH - 1 and baseCol >= 0 and baseCol < inW - 1):
                # 양선형 보간법을 사용하여 현재 픽셀의 값을 계산
                (interpolatedValue) = (1 - dx) * (1 - dy) * inImage[baseRow][baseCol] + \
                                      dx * (1 - dy) * inImage[baseRow + 1][baseCol] + \
                                      (1 - dx) * dy * inImage[baseRow][baseCol + 1] + \
                                      dx * dy * inImage[baseRow + 1][baseCol + 1]
                # 계산된 값으로 출력 이미지의 현재 픽셀을 설정합니다.
                outImage[i][k] = max(0, int(interpolatedValue))
            else:
                outImage[i][k] = 255
    displayImage()
    updateWindowTitle("확대(양선형 보간)")


def zoomout():  # 축소
    global window, canvas, paper, fullname
    global inImage, outImage, inH, inW, outH, outW

    # 정수 값 입력
    scale = askinteger('배율 입력', '정수 값 ', maxvalue=4)

    # 출력 이미지의 크기 결정
    outH = (int)(inH / scale)
    outW = (int)(inW / scale)

    # 출력 이미지 메모리 확보
    outImage = malloc2D(outH, outW)

    for i in range(inH):
        for k in range(inW):
            outImage[(int)(i / scale)][(int)(k / scale)] = inImage[i][k]

    displayImage()
    updateWindowTitle("축소")


def moveImage():  # 이동
    global window, canvas, paper, fullname
    global inImage, outImage, inH, inW, outH, outW

    # 정수 값 입력
    a = askinteger('x축', '정수 값 ')
    b = askinteger('y축', '정수 값')
    # 출력 이미지의 크기 결정
    outH = inH
    outW = inW

    # 출력 이미지 메모리 확보
    outImage = malloc2D(outH, outW)

    for i in range(inH):
        for k in range(inW):
            # 이미지 이동 시 경계를 벗어나는 경우를 고려하여 인덱스를 체크하여 처리
            if (i + a >= 0 and i + a < inH and k - b >= 0 and k - b < inW):
                outImage[i][k] = inImage[i + a][k - b]
            else:
                outImage[i][k] = 0  # 경계를 벗어나는 경우에는 0으로 처리하거나 다른 방법으로 처리
    displayImage()
    updateWindowTitle("이동")


def rotate1():  # 회전
    global window, canvas, paper, fullname
    global inImage, outImage, inH, inW, outH, outW

    # 정수 값 입력
    degree = askinteger('각도', '정수 값 ')
    radian = degree * 3.141592 / 180.0

    # 출력 이미지의 크기 결정
    outH = inH
    outW = inW

    # 원본 이미지 안나타나게
    outImage = [[0] * outW for _ in range(outH)]

    for i in range(inH):
        for k in range(inW):
            xs = i
            ys = k
            xd = int(cos(radian) * xs - sin(radian) * ys)
            yd = int(sin(radian) * xs + cos(radian) * ys)
            if ((0 <= xd and xd < outH) and (0 <= yd and yd < outW)):
                outImage[xd][yd] = inImage[xs][ys]
    displayImage()
    updateWindowTitle("회전")


def rotate2():  # 회전(중앙, 백워딩)
    global window, canvas, paper, fullname
    global inImage, outImage, inH, inW, outH, outW

    # 정수 값 입력
    degree = askinteger('각도', '정수 값 ')
    radian = degree * 3.141592 / 180.0

    # 출력 이미지의 크기 결정
    outH = inH
    outW = inW

    # 회전 중심점 계산
    cx = inH // 2
    cy = inW // 2

    # 원본 이미지 안나타나게
    outImage = [[0] * outW for _ in range(outH)]

    for i in range(outH):
        for k in range(outW):
            xd = i
            yd = k
            xs = int(cos(radian) * (xd - cx) + sin(radian) * (yd - cy))
            ys = int(-sin(radian) * (xd - cx) + cos(radian) * (yd - cy))
            xs += cx
            ys += cy
            if 0 <= xs < outH and 0 <= ys < outW:
                outImage[xd][yd] = inImage[xs][ys]
    displayImage()
    updateWindowTitle("회전(중앙,백워딩)")


def rotateZoom1():  # 회전 + 확대
    global window, canvas, paper, fullname
    global inImage, outImage, inH, inW, outH, outW

    # 정수 값 입력
    degree = askinteger('각도', '정수 값 ')
    scale = askinteger('배율', '정수 값 ')

    radian = -degree * 3.141592 / 180.0

    # 출력 이미지의 크기 결정
    outH = inH * scale
    outW = inW * scale
    # 출력 이미지 메모리 확보
    outImage = malloc2D(outH, outW)

    # 회전 중심점 계산
    cx = inH // 2
    cy = inW // 2

    # 원본 이미지 안나타나게
    outImage = [[0] * outW for _ in range(outH)]

    for i in range(outH):
        for k in range(outW):
            # 회전 및 확대된 픽셀 위치 계산
            xd = i // scale
            yd = k // scale
            xs = int(cos(radian) * (xd - cx) - sin(radian) * (yd - cy) + cx)
            ys = int(sin(radian) * (xd - cx) + cos(radian) * (yd - cy) + cy)

            # 회전된 픽셀 위치의 유효성 검사
            if 0 <= xs < inH and 0 <= ys < inW:
                outImage[i][k] = inImage[xs][ys]
            else:
                outImage[i][k] = 255  # 흰색으로 설정

    displayImage()
    updateWindowTitle("회전+확대")


def mirrorImage():
    global window, canvas, paper, fullname
    global inImage, outImage, inH, inW, outH, outW

    a = askinteger('정수', '상하:1 / 좌우:2 ')
    # 출력 이미지의 크기 결정
    outH = inH
    outW = inW

    # 출력 이미지 메모리 확보
    outImage = malloc2D(outH, outW)

    # 1일때는 상하, 2일때는 좌우 반전
    for i in range(outH):
        for k in range(outW):
            if a == 1:
                outImage[i][k] = inImage[inH - 1 - i][k]  # 상하 반전
            else:
                outImage[i][k] = inImage[i][inW - 1 - k]  # 좌우 반전

    # 출력하는 문자열 선택
    mirrorType = "상하 대칭" if a == 1 else "좌우 대칭"
    displayImage()
    updateWindowTitle("대칭")


#### 화소 영역 처리

def emboss():  # 엠보싱
    global window, canvas, paper, fullname
    global inImage, outImage, inH, inW, outH, outW

    # 출력 이미지의 크기 결정
    outH = inH
    outW = inW
    # 출력 이미지 메모리 확보
    outImage = malloc2D(outH, outW)

    # 엠보싱 마스크 정의
    mask = [[-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]]

    # 임시 입력 이미지 초기화 (패딩 추가)
    tmpInImage = [[127] * (inW + 2) for _ in range(inH + 2)]
    for i in range(inH):
        for k in range(inW):
            tmpInImage[i + 1][k + 1] = inImage[i][k]

    # 회선 연산
    tmpOutImage = [[0] * outW for _ in range(outH)]
    for i in range(inH):
        for k in range(inW):
            S = 0.0
            for m in range(3):
                for n in range(3):
                    S += tmpInImage[i + m][k + n] * mask[m][n]
            tmpOutImage[i][k] = S

    # 후처리
    for i in range(outH):
        for k in range(outW):
            tmpOutImage[i][k] += 127.0
            if tmpOutImage[i][k] < 0.0:
                outImage[i][k] = 0
            elif tmpOutImage[i][k] > 255.0:
                outImage[i][k] = 255
            else:
                outImage[i][k] = int(tmpOutImage[i][k])
    displayImage()
    updateWindowTitle("엠보싱")


def blur():  # 블러
    global window, canvas, paper, fullname
    global inImage, outImage, inH, inW, outH, outW

    # 출력 이미지의 크기 결정
    outH = inH
    outW = inW
    # 출력 이미지 메모리 확보
    outImage = malloc2D(outH, outW)

    # 블러링 마스크 정의
    mask = [[1 / 16, 2 / 16, 1 / 16],
            [2 / 16, 4 / 16, 2 / 16],
            [1 / 16, 2 / 16, 1 / 16]]

    # 임시 입력 이미지 초기화 (패딩 추가)
    tmpInImage = [[127] * (inW + 2) for _ in range(inH + 2)]
    for i in range(inH):
        for k in range(inW):
            tmpInImage[i + 1][k + 1] = inImage[i][k]

    # 회선 연산
    tmpOutImage = [[0] * outW for _ in range(outH)]
    for i in range(inH):
        for k in range(inW):
            S = 0.0
            for m in range(3):
                for n in range(3):
                    S += tmpInImage[i + m][k + n] * mask[m][n]
            tmpOutImage[i][k] = S

    # 후처리
    for i in range(outH):
        for k in range(outW):
            if tmpOutImage[i][k] < 0.0:
                outImage[i][k] = 0
            elif tmpOutImage[i][k] > 255.0:
                outImage[i][k] = 255
            else:
                outImage[i][k] = int(tmpOutImage[i][k])
    displayImage()
    updateWindowTitle("블러")


def sharp():
    global window, canvas, paper, fullname
    global inImage, outImage, inH, inW, outH, outW

    # 출력 이미지의 크기 결정
    outH = inH
    outW = inW
    # 출력 이미지 메모리 확보
    outImage = malloc2D(outH, outW)

    # 샤프닝 마스크 정의
    mask = [[0.0, -1.0, 0.0],
            [-1.0, 5.0, -1.0],
            [0.0, -1.0, 0.0]]

    # 임시 입력 이미지 초기화 (패딩 추가)
    tmpInImage = [[127] * (inW + 2) for _ in range(inH + 2)]
    for i in range(inH):
        for k in range(inW):
            tmpInImage[i + 1][k + 1] = inImage[i][k]

    # 회선 연산
    tmpOutImage = [[0] * outW for _ in range(outH)]
    for i in range(inH):
        for k in range(inW):
            S = 0.0
            for m in range(3):
                for n in range(3):
                    S += tmpInImage[i + m][k + n] * mask[m][n]
            tmpOutImage[i][k] = S

    # 후처리
    for i in range(outH):
        for k in range(outW):
            if tmpOutImage[i][k] < 0.0:
                outImage[i][k] = 0
            elif tmpOutImage[i][k] > 255.0:
                outImage[i][k] = 255
            else:
                outImage[i][k] = int(tmpOutImage[i][k])
    displayImage()
    updateWindowTitle("샤프닝")


def edge1():
    global window, canvas, paper, fullname
    global inImage, outImage, inH, inW, outH, outW

    # 출력 이미지의 크기 결정
    outH = inH
    outW = inW
    # 출력 이미지 메모리 확보
    outImage = malloc2D(outH, outW)
    # 수직 에지 검출 마스크 정의
    mask = [[0.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0]]

    # 임시 입력 이미지 초기화 (패딩 추가)
    tmpInImage = [[127] * (inW + 2) for _ in range(inH + 2)]
    for i in range(inH):
        for k in range(inW):
            tmpInImage[i + 1][k + 1] = inImage[i][k]

    # 회선 연산
    tmpOutImage = [[0] * outW for _ in range(outH)]
    for i in range(inH):
        for k in range(inW):
            S = 0.0
            for m in range(3):
                for n in range(3):
                    S += tmpInImage[i + m][k + n] * mask[m][n]
            tmpOutImage[i][k] = S

    # 후처리
    for i in range(outH):
        for k in range(outW):
            if tmpOutImage[i][k] < 0.0:
                outImage[i][k] = 0
            elif tmpOutImage[i][k] > 255.0:
                outImage[i][k] = 255
            else:
                outImage[i][k] = int(tmpOutImage[i][k])
    displayImage()
    updateWindowTitle("경계선 처리")


## 전역 변수부
window, canvas, paper = None, None, None
inImage, outImage = [], []
inH, inW, outH, outW = [0] * 4
fullname = ''

## 메인부
window = Tk()  # 벽
window.geometry("500x500")
window.resizable(width=False, height=False)
window.title("영상 처리 (RC 1)")

canvas = Canvas(window, height=256, width=256, bg='yellow')  # 칠판
paper = PhotoImage(height=256, width=256)  # 종이
canvas.create_image((256 // 2, 256 // 2), image=paper, state='normal')

# 메뉴 만들기
mainMenu = Menu(window)  # 메뉴의 틀
window.config(menu=mainMenu)

fileMenu = Menu(mainMenu, tearoff=0)  # 상위 메뉴 (파일)
mainMenu.add_cascade(label='파일', menu=fileMenu)
fileMenu.add_command(label='열기', command=openImage)
fileMenu.add_command(label='저장', command=saveImage)
fileMenu.add_separator()
fileMenu.add_command(label='종료', command=None)

pixelMenu = Menu(mainMenu)  # 상위 메뉴 (화소 점 처리)
mainMenu.add_cascade(label='화소점 처리', menu=pixelMenu)
pixelMenu.add_command(label='동일 이미지', command=equalImage)
pixelMenu.add_command(label='밝게/어둡게', command=brightImage)
pixelMenu.add_command(label='흑백', command=grayImage)
pixelMenu.add_command(label='AND', command=andImage)
pixelMenu.add_command(label='OR', command=orImage)
pixelMenu.add_command(label='XOR', command=xorImage)
pixelMenu.add_command(label='감마', command=gammaImage)
pixelMenu.add_command(label='파라볼라 CAP', command=parabolCapImage)
pixelMenu.add_command(label='파라볼라 CUP', command=parabolCupImage)
pixelMenu.add_command(label='반전', command=reverseImage)

histogramMenu = Menu(mainMenu)  # 상위 메뉴 (히스토그램 처리)
mainMenu.add_cascade(label='히스토그램 처리', menu=histogramMenu)
histogramMenu.add_command(label='히스토그램 스트레칭', command=histoStretch)
histogramMenu.add_command(label='평활화', command=histoEqual)
histogramMenu.add_command(label='엔드-인', command=endIn)

geometryMenu = Menu(mainMenu)  # 상위 메뉴 (히스토그램 처리)
mainMenu.add_cascade(label='기하학적 처리', menu=geometryMenu)
geometryMenu.add_command(label='확대(포워딩)', command=zoomIn1)
geometryMenu.add_command(label='확대(백워딩)', command=zoomIn2)
geometryMenu.add_command(label='확대(양선형 보간)', command=zoomIn3)
geometryMenu.add_command(label='축소', command=zoomout)
geometryMenu.add_command(label='이동', command=moveImage)
geometryMenu.add_command(label='회전', command=rotate1)
geometryMenu.add_command(label='회전(중앙, 백워딩)', command=rotate2)
geometryMenu.add_command(label='회전(중앙, 백워딩, 확대)', command=rotateZoom1)
geometryMenu.add_command(label='대칭', command=mirrorImage)

pixelareaMenu = Menu(mainMenu)  # 상위 메뉴 (화소 영역 처리)
mainMenu.add_cascade(label='화소 영역 처리', menu=pixelareaMenu)
pixelareaMenu.add_command(label='엠보싱', command=emboss)
pixelareaMenu.add_command(label='블러', command=blur)
pixelareaMenu.add_command(label='샤프닝', command=sharp)
pixelareaMenu.add_command(label='경계선 검출', command=edge1)

# editMenu = Menu(mainMenu)  # 상위 메뉴 (파일)
# mainMenu.add_cascade(label='편집', menu=editMenu)
# editMenu.add_command(label='복사', command=copyFile)
# editMenu.add_command(label='잘라내기', command=copyFile)
# editMenu.add_command(label='붙여넣기', command=copyFile)


# 컨트롤 == 위젯
# label1 = Label(window, text='나는 글자다', font=('궁서체',20), fg='red', bg='yellow' )
# button1 = Button(window, text='나를 클릭해줘~')
#
#
# label1.pack()
# button1.pack(side=BOTTOM)

canvas.pack()
window.mainloop()
