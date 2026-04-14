import os

from transformers import AutoTokenizer, BertModel, RobertaModel


def _bert_pretrained_src():
    """Same logic as utils.sgnav_bert_pretrained (env only); inlined so GroundingDINO never needs ``import utils``."""
    p = os.environ.get("SGNAV_BERT_BASE_UNCASED_PATH", "").strip()
    if not p:
        return "bert-base-uncased"
    ap = os.path.abspath(os.path.expanduser(p))
    return ap if os.path.isdir(ap) else "bert-base-uncased"


def get_tokenlizer(text_encoder_type, bert_base_uncased_path):
    if not isinstance(text_encoder_type, str):
        if hasattr(text_encoder_type, "text_encoder_type"):
            text_encoder_type = text_encoder_type.text_encoder_type
        elif text_encoder_type.get("text_encoder_type", False):
            text_encoder_type = text_encoder_type.get("text_encoder_type")
        else:
            raise ValueError(
                "Unknown type of text_encoder_type: {}".format(type(text_encoder_type))
            )

    if text_encoder_type == "bert-base-uncased":
        if is_bert_model_use_local_path(bert_base_uncased_path):
            print("use local bert model path: {}".format(bert_base_uncased_path))
            return AutoTokenizer.from_pretrained(bert_base_uncased_path)
        src = _bert_pretrained_src()
        if src != "bert-base-uncased":
            print("use local bert model path: {}".format(src))
        else:
            print("final text_encoder_type: {}".format(text_encoder_type))
        return AutoTokenizer.from_pretrained(src)

    print("final text_encoder_type: {}".format(text_encoder_type))
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)
    return tokenizer


def get_pretrained_language_model(text_encoder_type, bert_base_uncased_path):
    if text_encoder_type == "bert-base-uncased":
        if is_bert_model_use_local_path(bert_base_uncased_path):
            return BertModel.from_pretrained(bert_base_uncased_path)
        return BertModel.from_pretrained(_bert_pretrained_src())
    if text_encoder_type == "roberta-base":
        return RobertaModel.from_pretrained(text_encoder_type)
    raise ValueError("Unknown text_encoder_type {}".format(text_encoder_type))


def is_bert_model_use_local_path(bert_base_uncased_path):
    return bert_base_uncased_path is not None and len(bert_base_uncased_path) > 0
