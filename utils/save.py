
import pandas as pd
import pickle
#from utils.define import TRAIN_RESULT_PATH, VALID_RESULT_PATH, TRAIN_STEP_RESULT_PATH, logger

def save_epoch_result(train_result, valid_result):
    """ save result of training (unit : epoch) """
    train_dict, train_loss, train_cer = train_result
    valid_dict, valid_loss, valid_cer = valid_result
    train_dict["loss"].append(train_loss)
    train_dict["cer"].append(train_cer)
    valid_dict["loss"].append(valid_loss)
    valid_dict["cer"].append(valid_cer)

    train_df = pd.DataFrame(train_dict)
    valid_df = pd.DataFrame(valid_dict)
    train_df.to_csv(TRAIN_RESULT_PATH, encoding="cp949", index=False)
    valid_df.to_csv(VALID_RESULT_PATH, encoding="cp949", index=False)

def save_step_result(train_step_result, loss, cer):
    """ save result of training (unit : K time step) """
    train_step_result["loss"].append(loss)
    train_step_result["cer"].append(cer)
    train_step_df = pd.DataFrame(train_step_result)
    train_step_df.to_csv(TRAIN_STEP_RESULT_PATH, encoding="cp949", index=False)

def save_pickle(save_var, savepath, message=""):
    """ save pickle file """
    with open(savepath, "wb") as f:
        pickle.dump(save_var, f)
    logger.info(message)