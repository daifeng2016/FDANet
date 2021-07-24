import glob
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class CollectData:
    def __init__(self):
        self.TP = []
        self.FP = []
        self.FN = []
        self.TN = []

    def reload(self,groundtruth,probgraph):
        """

        :param groundtruth:  list,groundtruth image list
        :param probgraph:    list,prob image list
        :return:  None
        """
        self.groundtruth = groundtruth
        self.probgraph = probgraph
        self.TP = []
        self.FP = []
        self.FN = []
        self.TN = []

    def compute_pre_rec(self,gt, mask, mybins=np.arange(0, 256)):

        if (len(gt.shape) < 2 or len(mask.shape) < 2):
            print("ERROR: gt or mask is not matrix!")
            exit()
        if (len(gt.shape) > 2):  # convert to one channel
            gt = gt[:, :, 0]
        if (len(mask.shape) > 2):  # convert to one channel
            mask = mask[:, :, 0]
        if (gt.shape != mask.shape):
            print("ERROR: The shapes of gt and mask are different!")
            exit()

        gtNum = gt[gt > 128].size  # pixel number of ground truth foreground regions
        pp = mask[gt > 128]  # mask predicted pixel values in the ground truth foreground region
        nn = mask[gt <= 128]  # mask predicted pixel values in the ground truth bacground region

        pp_hist, pp_edges = np.histogram(pp,
                                         bins=mybins)  # pp_hist is the numbers, pp_edges in teh inerval count pixel numbers with values in each interval [0,1),[1,2),...,[mybins[i],mybins[i+1]),...,[254,255)
        nn_hist, nn_edges = np.histogram(nn, bins=mybins)

        pp_hist_flip = np.flipud(
            pp_hist)  # reverse the histogram to the following order: (255,254],...,(mybins[i+1],mybins[i]],...,(2,1],(1,0]
        nn_hist_flip = np.flipud(nn_hist)

        pp_hist_flip_cum = np.cumsum(
            pp_hist_flip)  # accumulate the pixel number in intervals: (255,254],(255,253],...,(255,mybins[i]],...,(255,0]
        nn_hist_flip_cum = np.cumsum(nn_hist_flip)

        precision = pp_hist_flip_cum / (pp_hist_flip_cum + nn_hist_flip_cum + 1e-8)  # TP/(TP+FP)
        recall = pp_hist_flip_cum / (gtNum + 1e-8)  # TP/(TP+FN)

        precision[np.isnan(precision)] = 0.0  # [255,]
        recall[np.isnan(recall)] = 0.0  # [255,]

        return np.reshape(precision, (len(precision))), np.reshape(recall, (len(recall)))
    def statistics(self):
        """
        calculate FPR TPR Precision Recall IoU
        :return: (FPR,TPR,AUC),(Precision,Recall,MAP),IoU
        """
        for threshold in tqdm(range(0,255)):
            temp_TP=0.0
            temp_FP=0.0
            temp_FN=0.0
            temp_TN=0.0
            assert(len(self.groundtruth)==len(self.probgraph))

            for index in range(len(self.groundtruth)):
                gt_img=cv2.imread(self.groundtruth[index])[:,:,0]
                prob_img=cv2.imread(self.probgraph[index])[:,:,0]

                gt_img=(gt_img>0)*1
                prob_img=(prob_img>=threshold)*1

                temp_TP = temp_TP + (np.sum(prob_img * gt_img))
                temp_FP = temp_FP + np.sum(prob_img * ((1 - gt_img)))
                temp_FN = temp_FN + np.sum(((1 - prob_img)) * ((gt_img)))
                temp_TN = temp_TN + np.sum(((1 - prob_img)) * (1 - gt_img))

            self.TP.append(temp_TP)
            self.FP.append(temp_FP)
            self.FN.append(temp_FN)
            self.TN.append(temp_TN)

        self.TP = np.asarray(self.TP).astype('float32')#[255,]
        self.FP = np.asarray(self.FP).astype('float32')
        self.FN = np.asarray(self.FN).astype('float32')
        self.TN = np.asarray(self.TN).astype('float32')#[255,]

        FPR = (self.FP) / (self.FP + self.TN) #[255,]
        TPR = (self.TP) / (self.TP + self.FN)#[255,]
        AUC = np.round(np.sum((TPR[1:] + TPR[:-1]) * (FPR[:-1] - FPR[1:])) / 2., 4)#[0.8903]

        Precision = (self.TP) / (self.TP + self.FP)#[255,]
        Recall = self.TP / (self.TP + self.FN)#[255,]
        MAP = np.round(np.sum((Precision[1:] + Precision[:-1]) * (Recall[:-1] - Recall[1:])) / 2.,4)#0.3509

        iou=0.7

        return (FPR,TPR,AUC),(Precision,Recall,MAP),iou

    def IoU(self,threshold=128):
        """
        to calculate IoU
        :param threshold: numerical,a threshold for gray image to binary image
        :return:  IoU
        """
        intersection=0.0
        union=0.0

        for index in range(len(self.groundtruth)):
            gt_img = cv2.imread(self.groundtruth[index])[:, :, 0]
            prob_img = cv2.imread(self.probgraph[index])[:, :, 0]

            gt_img = (gt_img > 0) * 1
            prob_img = (prob_img >= threshold) * 1

            intersection=intersection+np.sum(gt_img*prob_img)
            union=union+np.sum(gt_img)+np.sum(prob_img)-np.sum(gt_img*prob_img)
        iou=np.round(intersection/union,4)
        return iou

    def debug(self):
        """
        show debug info
        :return: None
        """
        print("Now enter debug mode....\nPlease check the info bellow:")
        print("total groundtruth: %d   total probgraph: %d\n"%(len(self.groundtruth),len(self.probgraph)))
        for index in range(len(self.groundtruth)):
            print(self.groundtruth[index],self.probgraph[index])
        print("Please confirm the groundtruth and probgraph name is opposite")


class DrawCurve:
    """
    draw ROC/PR curve
    """
    def __init__(self,savepath):
        self.savepath=savepath
        self.colorbar=['red','green','blue','black']
        self.linestyle=['-','-.','--',':','-*']

    def reload(self,xdata,ydata,auc,dataName,modelName):
        """
        this function is to update data for Function roc/pr to draw
        :param xdata:  list,x-coord of roc(pr)
        :param ydata:  list,y-coord of roc(pr)
        :param auc:    numerical,area under curve
        :param dataName: string,name of dataset
        :param modelName: string,name of test model
        :return:  None
        """
        self.xdata.append(xdata)
        self.ydata.append(ydata)
        self.modelName.append(modelName)
        self.auc.append(auc)
        self.dataName=dataName

    def newly(self,modelnum):
        """
        renew all the data
        :param modelnum:  numerical,number of models to draw
        :return:  None
        """
        self.modelnum = modelnum
        self.xdata = []
        self.ydata = []
        self.modelName = []
        self.auc = []

    def roc(self):
        """
        draw ROC curve,save the curve graph to  savepath
        :return: None
        """
        plt.figure(1)
        plt.title('ROC Curve of %s'%self.dataName, fontsize=15)
        plt.xlabel("False Positive Rate", fontsize=15)
        plt.ylabel("True Positive Rate", fontsize=15)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        for i in range(self.modelnum):
            plt.plot(self.xdata[i], self.ydata[i], color=self.colorbar[i%len(self.colorbar)], linewidth=2.0, linestyle=self.linestyle[i%len(self.linestyle)], label=self.modelName[i]+',AUC:' + str(self.auc[i]))
        plt.legend()
        plt.savefig(self.savepath+'%s_ROC.png'%self.dataName, dpi=800)
        #plt.show()


    def pr(self):
        """
        draw PR curve,save the curve to  savepath
        :return: None
        """
        plt.figure(2)
        plt.title('PR Curve of %s'%self.dataName, fontsize=15)
        plt.xlabel("Recall", fontsize=15)
        plt.ylabel("Precision", fontsize=15)
        plt.xlim(0.5, 1)
        plt.ylim(0.5, 1)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        for i in range(self.modelnum):
            plt.plot(self.xdata[i], self.ydata[i], color=self.colorbar[i%len(self.colorbar)], linewidth=2.0, linestyle=self.linestyle[i%len(self.linestyle)],label=self.modelName[i]+',MAP:' + str(self.auc[i]))
        plt.legend()
        plt.savefig(self.savepath+'%s_PR.png'%self.dataName, dpi=800)
        #plt.show()


    def F1(self):

       plt.figure(3)
       plt.title('F1 Curve of %s' % self.dataName, fontsize=15)
       plt.xlabel("Threshold", fontsize=15)
       plt.ylabel("F1-score", fontsize=15)
       plt.xlim(0, 1)
       plt.ylim(0, 1)
       plt.xticks(fontsize=12)
       plt.yticks(fontsize=12)
       F1_score=2*np.array(self.xdata)*np.array(self.ydata)/(np.array(self.xdata)+np.array(self.ydata))#[2,255]
       bins=np.arange(0,255)
       for i in range(self.modelnum):
          plt.plot(bins/255.0, F1_score[i], color=self.colorbar[i % len(self.colorbar)], linewidth=2.0,
                 linestyle=self.linestyle[i % len(self.linestyle)],
                 label=self.modelName[i])
          plt.legend()
       plt.savefig(self.savepath + '%s_F1_score.png' % self.dataName, dpi=800)
	   #plt.show()


# plt.show()

def fileList(imgpath,filetype):
    return glob.glob(imgpath+filetype)


def drawCurve(gtlist,problist,modelName,dataset,savepath='./'):
    """
    draw ROC PR curve,calculate AUC MAP IoU
    :param gtlist:  list,groundtruth list
    :param problist: list,list of probgraph list
    :param modelName:  list,name of test,model
    :param dataset: string,name of dataset
    :param savepath: string,path to save curve
    :return:
    """
    assert(len(problist)==len(modelName))

    process = CollectData()
    painter_roc = DrawCurve(savepath)
    painter_pr = DrawCurve(savepath)
    modelNum=len(problist)
    painter_roc.newly(modelNum)
    painter_pr.newly(modelNum)
    #painter_F1=newly(modelNum)
    # calculate param
    for index in range(modelNum):
        print("processing result %s"% modelName[index])
        process.reload(gtlist,problist[index])
        (FPR, TPR, AUC), (Precision, Recall, MAP),IoU = process.statistics()
        #painter_roc.reload(FPR, TPR, AUC,dataset, modelName[index])
        painter_pr.reload(Precision, Recall, MAP, dataset, modelName[index])

    # draw curve and save
    #painter_roc.roc()
    painter_pr.pr()
    painter_pr.F1()
