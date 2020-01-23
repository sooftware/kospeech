import pandas as pd

def save_csv(train_result, valid_result):
    train_dict, train_loss, train_cer = train_result
    valid_dict, valid_loss, valid_cer = valid_result
    train_dict["loss"].append(train_loss)
    train_dict["cer"].append(train_cer)
    valid_dict["loss"].append(valid_loss)
    valid_dict["cer"].append(valid_cer)

    train_df = pd.DataFrame(train_dict)
    valid_df = pd.DataFrame(valid_dict)
    train_df.to_csv("../csv/train_result.csv", encoding='cp949', index=False)
    valid_df.to_csv("../csv/eval_result.csv", encoding='cp949', index=False)

    del train_df, valid_df  # memory deallocation