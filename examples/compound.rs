use tch::{Device, Kind, Reduction, Tensor};

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct Example {
    #[serde(with = "tch_serde::serde_tensor")]
    tensor: Tensor,
    #[serde(with = "tch_serde::serde_kind")]
    kind: Kind,
    #[serde(with = "tch_serde::serde_device")]
    device: Device,
    #[serde(with = "tch_serde::serde_reduction")]
    reduction: Reduction,
}

fn main() {
    let example = Example {
        tensor: Tensor::randn(&[2, 3], (Kind::Float, Device::cuda_if_available())),
        kind: Kind::Float,
        device: Device::Cpu,
        reduction: Reduction::Mean,
    };
    let text = serde_json::to_string_pretty(&example).unwrap();
    println!("{}", text);
}
