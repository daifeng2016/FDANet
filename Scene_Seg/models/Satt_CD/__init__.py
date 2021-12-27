import logging
logger = logging.getLogger('base')
from utils.utils import print_model_para,print_model_parm_nums
def create_model(opt):
    model = opt['model']

    if model == 'BRSeg':
        from .BRSeg_Model import BRSeg_Model as M
    elif model=='BRCD':
        from .CD_Model import CD_Model as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    if m.netG:
        print("model G parameters...")
        model_para = print_model_parm_nums(m.netG)
        message = 'model G parameters is {:.2f}M'.format(model_para)
        logger.info(message)

    if m.netD_seg:
        print("model D_seg parameters...")
        model_para = print_model_parm_nums(m.netD_seg)
        message = 'model D_seg parameters is {:.2f}M'.format(model_para)
        logger.info(message)

    if m.netD_fea:
        print("model D_fea parameters...")

        model_para = print_model_parm_nums(m.netD_fea)
        message = 'model D_fea parameters is {:.2f}M'.format(model_para)
        logger.info(message)

    if opt['train']['train_style'] and opt['DA_method']=="Atk":
        #print_model_parm_nums(m.netG_atk)
        model_para = print_model_parm_nums(m.netG_atk)
        message = 'model netG_atk parameters is {:.2f}M'.format(model_para)
        logger.info(message)
    if opt['train']['train_style'] and opt['DA_method']=="AdaIN":

        model_para = print_model_parm_nums(m.netG_AdaIN)
        message = 'model netG_AdaIN parameters is {:.2f}M'.format(model_para)
        logger.info(message)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m