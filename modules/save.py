"""
Copyright 2020- Kai.Lib
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pandas as pd
import pickle
from modules.define import logger

def save_epoch_result(train_result, valid_result):
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

def save_step_result(train_step_result, loss, cer):
    train_step_result["loss"].append(loss)
    train_step_result["cer"].append(cer)
    train_step_df = pd.DataFrame(train_step_result)
    train_step_df.to_csv("./csv/train_step_result.csv", encoding='cp949', index=False)

def save_pickle(save_var, savepath, message=""):
    with open(savepath, "wb") as f:
        pickle.dump(save_var, f)
    logger.info(message)