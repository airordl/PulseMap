import numpy as np
from matplotlib import pyplot as plt
import cv2
import io
import time
import math
from scipy.signal import savgol_filter

# Camera stream
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
cap.set(cv2.CAP_PROP_FPS, 30)
# Video stream (optional, not tested)
# cap = cv2.VideoCapture("video.mp4")
# Image crop
w, h =  10, 10

#x_min = 0
#x_max = 750
x_min = 200
x_max = 700
y_min = 200
y_max = 700
x = range (x_min, x_max,1* w)
y = range (y_min,y_max,1* h)


x_ = np.array(x)
y_ = np.array(y)

nofx = x_.shape[0]
nofy = y_.shape[0]


import sys

nofcrops = x_.shape[0] * y_.shape[0]

heartbeat_count = 128

HBV = []
HBT = []

now = [time.time()]
for i in range (nofcrops):
    HBV.append([0]*heartbeat_count)
    HBT.append(now*heartbeat_count)
now = time.time()
#HBV = np.array(HBV)
#HBT = np.array(HBT)

#heartbeat_values = [0]*heartbeat_count

#heartbeat_times = [time.time()]*heartbeat_count

time_zero = np.zeros(nofcrops)
time_zero += time.time()
sampling_time = time_zero



#sys.exit()

# Matplotlib graph surface
#fig = plt.figure()
#ax = fig.add_subplot(111)

fig, axs = plt.subplots(2,figsize = (15,9))


PLOTS = False
PLOTAVG = True
PLOTVEIN = True

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    X = img.shape[1]
    Y = img.shape[0]

    BPM_ = []
    FOURIER = []
    FREQ = []
    TSCALE = []
    TIMEV = [] 
    I = -1
    now = time.time()
    for x in x_:
        for y in y_:
            I+=1
            

            crop_img = img[y:y + h, x:x + w]
            # Update the data
            HBV[I] = (HBV[I])[1:] + [np.average(crop_img)]
            HBT[I] = (HBT[I])[1:] + [time.time()]
            TIMEV.append(HBV[I])

            freq =  (np.fft.rfft(HBV[I]))

            t_scale = np.array (HBT[I]) - time_zero[I]
            TSCALE = t_scale

            t_scale -= t_scale[0] #- 10**(-10)# expedient not to devide by 0
            t_max = t_scale[-1]
            
#            sampling_time[I] = time.time() - time_zero[I]
#
#            print (time.time() - time_zero[I])
#            time_zero[I] = time.time()
            f_sampling = 1./sampling_time[I]
            sampling_t = HBT[I][-1] - HBT[I][-2]
            f_sampling = 1./sampling_t
            n = np.array(HBV[I] ).shape[0]

            fstep = f_sampling/n
            f_scale = np.arange(freq.shape[0]) * fstep
            
            f_min = 5
            f_scale = f_scale[f_min:]
            freq = freq[f_min:]
            
            if PLOTAVG:
                FOURIER.append(freq)
                FREQ.append(f_scale)

            if PLOTS and I%5 ==0: 
                axs[0].plot(f_scale,abs(freq) )
                axs[1].plot(t_scale,HBV[I])
            #    axs[0].set_yscale('log')
                
                axs[0].set_xlabel('Hz')
                axs[1].set_xlabel('sec')
            
            fmax = np.amax(abs(freq))

#            if not PLOTAVG :
            for i in range (freq.shape[0]):
                if fmax == abs(freq[i]):
                    if PLOTS and I%5 == 0:
                        if not PLOTAVG:
                            axs[0].set_title('BPM = ' + str(round(f_scale[i]*60,0))+ ' var = '+str(sampling_t))
                    BPM_.append(f_scale[i] * 60)
                    break


    if PLOTAVG: 
        FOURIER = np.array(FOURIER)
        FREQ = np.array(FREQ)
        TIMEV = np.array(TIMEV)

        TIMEV = np.average(TIMEV, axis = 0)
        FOURIER = np.average(FOURIER,axis = 0)
        FREQ = np.average(FREQ, axis = 0)
        axs[0].plot(FREQ,abs(FOURIER) )
        axs[1].plot(TSCALE,TIMEV)
        axs[0].set_xlabel('Hz')
        axs[1].set_xlabel('sec')

        fmax = np.amax(abs(FOURIER))
        for i in range (FOURIER.shape[0]):
            if fmax == abs(FOURIER[i]):
                axs[0].set_title('BPM = ' + str(round(FREQ[i]*60,0)))


    
    if PLOTS or PLOTAVG:

        fig.canvas.draw()
        plt.cla()
        axs[0].cla()
        axs[1].cla()



        plot_img_np = np.frombuffer(fig.canvas.tostring_rgb(),
                                    dtype=np.uint8)
#        plot_img_np = np.fromstring(fig.canvas.tostring_rgb(),
#                                    dtype=np.uint8, sep='')
        plot_img_np = plot_img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # Display the frames
        
        IMG_MAX_VALUE = np.amax(img)
            # Draw matplotlib graph to numpy array
        if PLOTVEIN:
            BPM_ = np.array(BPM_)
            Q = -1
            for _ in BPM_:
                Q+=1
                if BPM_[Q] < 60 or BPM_[Q] > 95:
                    BPM_[Q] = 0
#                else:
#                    BPM_[Q] = IMG_MAX_VALUE
#            BPM_ = BPM_ / np.amax(BPM_)
#            BPM_ = BPM_ * IMG_MAX_VALUE

            BPM_ =BPM_.reshape(( nofx, nofy))
        else:
            BPM_ = np.array(BPM_)
            Q = -1
            for _ in BPM_:
                Q+=1
                BPM_[Q] = 0
            BPM_ = BPM_.reshape((nofx, nofy))
        
        
        img = np.array(img)
        

        ones = np.ones(w*h)
        ones = ones.reshape((w,h))
#        BPM_grid = np.kron (ones,BPM_) #no!
        BPM_grid = np.kron (BPM_,ones) #yes!


        upper_zero = np.zeros((x_min,y_min))
        lower_zero = np.zeros((X-x_max , Y-y_max  ))
        
        def direct_sum (a,b):
            ds = np.zeros(np.add(a.shape,b.shape))
            ds[:a.shape[0],:a.shape[1]]=a
            ds[a.shape[0]:,a.shape[1]:]=b
            return ds
        
        #To Sum
        TS = direct_sum(upper_zero,BPM_grid)



        TS = direct_sum(TS,lower_zero)


        img = np.transpose( img) + TS
        img =np.transpose(img)
        img = img/ np.amax(img)
        

        cv2.imshow('Crop', img)
        cv2.imshow('Graph',plot_img_np)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
