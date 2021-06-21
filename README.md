# tch-serde: Serialize/Deserialize tch-rs types with serde

This crate provides {ser,de}ialization methods for tch-rs common types.

[docs.rs](https://docs.rs/tch-serde/) | [crates.io](https://crates.io/crates/tch-serde)

## Usage

For example, annotate `#[serde(with = "tch_serde::serde_tensor")]` attributes to enable serialization on tensor field.

```rust
use tch::{Device, Kind, Tensor};

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
        tensor: Tensor::randn(&[2, 3], (Kind::Float, Device::Cuda(0))),
        kind: Kind::Float,
        device: Device::Cpu,
        reduction: Reduction::Mean,
    };
    let text = serde_json::to_string_pretty(&example).unwrap();
    println!("{}", text);
}
```

## License

MIT license. See the [LICENSE](LICENSE) file.
