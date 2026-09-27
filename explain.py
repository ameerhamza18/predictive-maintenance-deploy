
def get_feature_names(preprocessor):
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = cat_encoder.get_feature_names_out(["type", "tool_wear_bucket"])
    num_names = preprocessor.transformers_[1][2]
    return list(cat_names) + list(num_names)