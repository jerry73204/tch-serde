# tch-serde: Serialize/Deserialize tch-rs types with serde

This crate provides {ser,de}ialization methods for tch-rs common types.

## Usage

Add this line to your `Cargo.toml` to work with this crate.

```toml
tch-serde = "0.1"
```

Annotate `#[serde(with = "tch_serde::serde_tensor")]` attributes to enable serialization on that field.

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
}

fn main() {
    let example = Example {
        tensor: Tensor::randn(&[2, 3], (Kind::Float, Device::cuda_if_available())),
        kind: Kind::Float,
        device: Device::Cpu,
    };
    let text = serde_json::to_string_pretty(&example).unwrap();
    println!("{}", text);
}
```

## License

MIT license. See the [LICENSE](LICENSE) file.
