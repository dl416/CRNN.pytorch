import torch
from IIIT5K import IIIT5K

def prob_string(prob):
    dataset = IIIT5K()
    id_to_char = dataset.id_to_char
    if len(prob.shape) == 2:
        predict = ""
        for i in range(prob.shape[0]):
            _, predicted = torch.max(prob[i].data, 0)
            predict += id_to_char[predicted.item()]

        predict += "卍"
        string = ""
        for s1, s2 in zip(predict[:-2], predict[1:]):
            if s1 == "卍":
                continue
            elif s1 == s2:
                continue
            elif s1 != s2:
                string += s1

    if len(prob.shape) == 1:
        predict = ""
        for i in range(prob.shape[0]):
            predict += id_to_char[prob[i].item()]
        predict += "卍"
        string = ""
        for s1, s2 in zip(predict[:-2], predict[1:]):
            if s1 == "卍":
                continue
            elif s1 == s2:
                continue
            elif s1 != s2:
                string += s1
 
    return string

if __name__ == "__main__":
    prob = torch.randn(24, 63)
    
    prob_string(prob)