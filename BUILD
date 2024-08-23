package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "models",
    srcs = ["models.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//astronet/astro_cnn_model",
        "//astronet/astro_cnn_model:configurations",
        "//astronet/astro_fc_model",
        "//astronet/astro_fc_model:configurations",
        "//astronet/astro_model",
        "//astronet/astro_model:configurations",
        "//tf_util:configdict",
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":models",
        "//astronet/util:estimator_util",
        "//tf_util:config_util",
        "//tf_util:configdict",
        "//tf_util:estimator_runner",
    ],
)

py_binary(
    name = "evaluate",
    srcs = ["evaluate.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":models",
        "//astronet/util:estimator_util",
        "//tf_util:config_util",
        "//tf_util:configdict",
        "//tf_util:estimator_runner",
    ],
)

py_binary(
    name = "predict",
    srcs = ["predict.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":models",
        "//astronet/data:preprocess",
        "//astronet/util:estimator_util",
        "//tf_util:config_util",
        "//tf_util:configdict",
    ],
)
