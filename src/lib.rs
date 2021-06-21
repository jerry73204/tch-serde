//! Serialize/Deserialize [tch] types with [serde].
//!
//! The serializing and deserializing methods are groupped in `serde_tensor`,
//! `serde_kind` and other similar modules. You can annotate `#[serde(with = "tch_serde::serde_tensor")]`
//! attributes on fields to enable serialization.
//!
//! The snipplet serializes a compound type of [Tensor], [Kind] and [Device].
//!
//! ``` rust
//! use tch::{Tensor, Device, Kind, Reduction};
//!
//! #[derive(Debug, serde::Serialize, serde::Deserialize)]
//! struct Example {
//!     #[serde(with = "tch_serde::serde_tensor")]
//!     tensor: Tensor,
//!         #[serde(with = "tch_serde::serde_kind")]
//!     kind: Kind,
//!         #[serde(with = "tch_serde::serde_device")]
//!     device: Device,
//!         #[serde(with = "tch_serde::serde_reduction")]
//!     reduction: Reduction,
//! }
//!
//! let example = Example {
//!     tensor: Tensor::randn(
//!         &[2, 3],
//!         (Kind::Float, Device::Cuda(0)),
//!     ),
//!     kind: Kind::Float,
//!     device: Device::Cpu,
//!     reduction: Reduction::Mean,
//! };
//! let text = serde_json::to_string_pretty(&example).unwrap();
//! println!("{}", text);
//! ```
//!
//! For example, it produces the following JSON text.
//! ```json
//! {
//!   "tensor": {
//!     "requires_grad": false,
//!     "device": "cuda:0",
//!     "shape": [
//!       2,
//!       3
//!     ],
//!     "kind": "float",
//!     "data": [
//!       182,
//!       59,
//!       207,
//!       190,
//!       12,
//!       195,
//!       95,
//!       62,
//!       123,
//!       68,
//!       200,
//!       191,
//!       242,
//!       98,
//!       231,
//!       190,
//!       108,
//!       94,
//!       225,
//!       62,
//!       56,
//!       45,
//!       3,
//!       190
//!     ]
//!   },
//!   "kind": "float",
//!   "device": "cpu",
//!   "reduction": "mean",
//! }
//! ```

use half::f16;
use serde::{
    de::Error as DeserializeError, ser::Error as SerializeError, Deserialize, Deserializer,
    Serialize, Serializer,
};
use std::{borrow::Cow, mem};
use tch::{Device, Kind, Reduction, Tensor};

/// The serialized representation of [Tensor].
///
/// The  [Tensor] is converted to this type during serialization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorRepr {
    pub requires_grad: bool,
    #[serde(with = "serde_device")]
    pub device: Device,
    pub shape: Vec<i64>,
    #[serde(with = "serde_kind")]
    pub kind: Kind,
    pub data: Vec<u8>,
}

/// Serializing/Deserializing functions for [Tensor].
pub mod serde_tensor {
    use super::*;

    pub fn serialize<S>(tensor: &Tensor, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let device = tensor.device();
        let requires_grad = tensor.requires_grad();
        let shape = tensor.size();
        let kind = tensor.kind();

        let data = {
            let numel = tensor.numel();
            let elem_size = match kind {
                Kind::Uint8 => mem::size_of::<u8>(),
                Kind::Int8 => mem::size_of::<i8>(),
                Kind::Int16 => mem::size_of::<i16>(),
                Kind::Int => mem::size_of::<i32>(),
                Kind::Int64 => mem::size_of::<i64>(),
                Kind::Half => mem::size_of::<f16>(),
                Kind::Float => mem::size_of::<f32>(),
                Kind::Double => mem::size_of::<f64>(),
                Kind::Bool => mem::size_of::<bool>(),
                Kind::QInt8 => mem::size_of::<i8>(),
                Kind::QUInt8 => mem::size_of::<u8>(),
                Kind::QInt32 => mem::size_of::<i32>(),
                Kind::BFloat16 => mem::size_of::<f16>(),
                _ => {
                    return Err(S::Error::custom(format!(
                        "tensor with kind {:?} is not supported yet",
                        kind
                    )));
                }
            };
            let buf_size = numel * elem_size;
            let mut buffer = vec![0u8; buf_size];
            tensor.copy_data_u8(&mut buffer, numel);
            buffer
        };

        let repr = TensorRepr {
            requires_grad,
            device,
            shape,
            kind,
            data,
        };

        repr.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Tensor, D::Error>
    where
        D: Deserializer<'de>,
    {
        let TensorRepr {
            requires_grad,
            device,
            shape,
            kind,
            data,
        } = Deserialize::deserialize(deserializer)?;

        let tensor = Tensor::of_data_size(&data, &shape, kind);
        let tensor = tensor.set_requires_grad(requires_grad);
        let tensor = tensor.to_device(device);

        Ok(tensor)
    }
}

/// Serializing/Deserializing functions for [Device].
pub mod serde_device {
    use super::*;

    pub fn serialize<S>(device: &Device, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let text = match device {
            Device::Cpu => "cpu".into(),
            Device::Cuda(n) => format!("cuda:{}", n),
        };
        serializer.serialize_str(&text)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Device, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = String::deserialize(deserializer)?;
        let device = match text.as_str() {
            "cpu" => Device::Cpu,
            other => {
                let index = (move || -> Option<_> {
                    let remaining = other.strip_prefix("cuda:")?;
                    let index: usize = remaining.parse().ok()?;
                    Some(index)
                })()
                .ok_or_else(|| D::Error::custom(format!("invalid device name {}", text)))?;

                Device::Cuda(index)
            }
        };

        Ok(device)
    }
}

/// Serializing/Deserializing functions for [Kind].
pub mod serde_kind {
    use super::*;

    pub fn serialize<S>(kind: &Kind, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use Kind::*;
        let text = match kind {
            Uint8 => "uint8",
            Int8 => "int8",
            Int16 => "int16",
            Int => "int",
            Int64 => "int64",
            Half => "half",
            Float => "float",
            Double => "double",
            ComplexHalf => "complex_half",
            ComplexFloat => "complex_float",
            ComplexDouble => "complex_double",
            Bool => "bool",
            QInt8 => "qint8",
            QUInt8 => "quint8",
            QInt32 => "qint32",
            BFloat16 => "bfloat16",
        };
        text.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Kind, D::Error>
    where
        D: Deserializer<'de>,
    {
        use Kind::*;
        let text = String::deserialize(deserializer)?;
        let kind = match text.as_str() {
            "uint8" => Uint8,
            "int8" => Int8,
            "int16" => Int16,
            "int" => Int,
            "int64" => Int64,
            "half" => Half,
            "float" => Float,
            "double" => Double,
            "complex_half" => ComplexHalf,
            "complex_float" => ComplexFloat,
            "complex_double" => ComplexDouble,
            "bool" => Bool,
            "qint8" => QInt8,
            "quint8" => QUInt8,
            "qint32" => QInt32,
            "bfloat16" => BFloat16,
            _ => return Err(D::Error::custom(format!(r#"invalid kind "{}""#, text))),
        };
        Ok(kind)
    }
}

/// Serializing/Deserializing functions for [Reduction].
pub mod serde_reduction {
    use super::*;

    pub fn serialize<S>(reduction: &Reduction, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let text: Cow<'_, str> = match reduction {
            Reduction::None => "none".into(),
            Reduction::Mean => "mean".into(),
            Reduction::Sum => "sum".into(),
            Reduction::Other(value) => format!("other:{}", value).into(),
        };
        text.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Reduction, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = String::deserialize(deserializer)?;

        let reduction = match &*text {
            "none" => Reduction::None,
            "mean" => Reduction::Mean,
            "sum" => Reduction::Sum,
            other => {
                let value = (move || -> Option<i64> {
                    let remaining = other.strip_prefix("other:")?;
                    let value: i64 = remaining.parse().ok()?;
                    Some(value)
                })()
                .ok_or_else(|| D::Error::custom(format!("invalid reduction '{}'", other)))?;
                Reduction::Other(value)
            }
        };

        Ok(reduction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    #[test]
    fn serde_reduction_test() -> Result<()> {
        #[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
        struct Example(#[serde(with = "serde_reduction")] Reduction);

        assert_eq!(
            serde_json::from_str::<Example>(r#""none""#)?.0,
            Reduction::None
        );
        assert_eq!(
            serde_json::from_str::<Example>(r#""mean""#)?.0,
            Reduction::Mean
        );
        assert_eq!(
            serde_json::from_str::<Example>(r#""sum""#)?.0,
            Reduction::Sum
        );
        assert_eq!(
            serde_json::from_str::<Example>(r#""other:3""#)?.0,
            Reduction::Other(3)
        );
        assert_eq!(
            serde_json::to_string(&Example(Reduction::None))?,
            r#""none""#
        );
        assert_eq!(
            serde_json::to_string(&Example(Reduction::Mean))?,
            r#""mean""#
        );
        assert_eq!(serde_json::to_string(&Example(Reduction::Sum))?, r#""sum""#);
        assert_eq!(
            serde_json::to_string(&Example(Reduction::Other(1)))?,
            r#""other:1""#
        );

        Ok(())
    }

    #[test]
    fn serde_device_test() -> Result<()> {
        #[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
        struct Example(#[serde(with = "serde_device")] Device);

        // serialize
        assert_eq!(serde_json::to_string(&Example(Device::Cpu))?, r#""cpu""#);
        assert_eq!(
            serde_json::to_string(&Example(Device::Cuda(0)))?,
            r#""cuda:0""#
        );
        assert_eq!(
            serde_json::to_string(&Example(Device::Cuda(1)))?,
            r#""cuda:1""#
        );

        // deserialize
        assert_eq!(
            serde_json::from_str::<Example>(r#""cpu""#)?,
            Example(Device::Cpu)
        );
        assert_eq!(
            serde_json::from_str::<Example>(r#""cuda:0""#)?,
            Example(Device::Cuda(0))
        );
        assert_eq!(
            serde_json::from_str::<Example>(r#""cuda:1""#)?,
            Example(Device::Cuda(1))
        );

        Ok(())
    }

    #[test]
    fn serde_kind_test() -> Result<()> {
        #[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
        struct Example(#[serde(with = "serde_kind")] Kind);

        // serialize
        assert_eq!(serde_json::to_string(&Example(Kind::Int))?, r#""int""#);
        assert_eq!(serde_json::to_string(&Example(Kind::Float))?, r#""float""#);
        assert_eq!(serde_json::to_string(&Example(Kind::Uint8))?, r#""uint8""#);
        assert_eq!(serde_json::to_string(&Example(Kind::Int8))?, r#""int8""#);
        assert_eq!(serde_json::to_string(&Example(Kind::Int16))?, r#""int16""#);
        assert_eq!(serde_json::to_string(&Example(Kind::Int))?, r#""int""#);
        assert_eq!(serde_json::to_string(&Example(Kind::Int64))?, r#""int64""#);
        assert_eq!(serde_json::to_string(&Example(Kind::Half))?, r#""half""#);
        assert_eq!(serde_json::to_string(&Example(Kind::Float))?, r#""float""#);
        assert_eq!(
            serde_json::to_string(&Example(Kind::Double))?,
            r#""double""#
        );
        assert_eq!(
            serde_json::to_string(&Example(Kind::ComplexHalf))?,
            r#""complex_half""#
        );
        assert_eq!(
            serde_json::to_string(&Example(Kind::ComplexFloat))?,
            r#""complex_float""#
        );
        assert_eq!(
            serde_json::to_string(&Example(Kind::ComplexDouble))?,
            r#""complex_double""#
        );
        assert_eq!(serde_json::to_string(&Example(Kind::Bool))?, r#""bool""#);
        assert_eq!(serde_json::to_string(&Example(Kind::QInt8))?, r#""qint8""#);
        assert_eq!(
            serde_json::to_string(&Example(Kind::QUInt8))?,
            r#""quint8""#
        );
        assert_eq!(
            serde_json::to_string(&Example(Kind::QInt32))?,
            r#""qint32""#
        );
        assert_eq!(
            serde_json::to_string(&Example(Kind::BFloat16))?,
            r#""bfloat16""#
        );

        // deserialize
        assert_eq!(
            serde_json::from_str::<Example>(r#""int""#)?,
            Example(Kind::Int)
        );
        assert_eq!(
            serde_json::from_str::<Example>(r#""float""#)?,
            Example(Kind::Float)
        );
        assert_eq!(
            serde_json::from_str::<Example>(r#""uint8""#)?,
            Example(Kind::Uint8)
        );
        assert_eq!(
            serde_json::from_str::<Example>(r#""int8""#)?,
            Example(Kind::Int8)
        );
        assert_eq!(
            serde_json::from_str::<Example>(r#""int16""#)?,
            Example(Kind::Int16)
        );
        assert_eq!(
            serde_json::from_str::<Example>(r#""int""#)?,
            Example(Kind::Int)
        );
        assert_eq!(
            serde_json::from_str::<Example>(r#""int64""#)?,
            Example(Kind::Int64)
        );
        assert_eq!(
            serde_json::from_str::<Example>(r#""half""#)?,
            Example(Kind::Half)
        );
        assert_eq!(
            serde_json::from_str::<Example>(r#""float""#)?,
            Example(Kind::Float)
        );
        assert_eq!(
            serde_json::from_str::<Example>(r#""double""#)?,
            Example(Kind::Double)
        );
        assert_eq!(
            serde_json::from_str::<Example>(r#""complex_half""#)?,
            Example(Kind::ComplexHalf)
        );
        assert_eq!(
            serde_json::from_str::<Example>(r#""complex_float""#)?,
            Example(Kind::ComplexFloat)
        );
        assert_eq!(
            serde_json::from_str::<Example>(r#""complex_double""#)?,
            Example(Kind::ComplexDouble)
        );
        assert_eq!(
            serde_json::from_str::<Example>(r#""bool""#)?,
            Example(Kind::Bool)
        );
        assert_eq!(
            serde_json::from_str::<Example>(r#""qint8""#)?,
            Example(Kind::QInt8)
        );
        assert_eq!(
            serde_json::from_str::<Example>(r#""quint8""#)?,
            Example(Kind::QUInt8)
        );
        assert_eq!(
            serde_json::from_str::<Example>(r#""qint32""#)?,
            Example(Kind::QInt32)
        );
        assert_eq!(
            serde_json::from_str::<Example>(r#""bfloat16""#)?,
            Example(Kind::BFloat16)
        );

        Ok(())
    }

    #[test]
    fn serde_tensor() -> Result<()> {
        #[derive(Debug, Serialize, Deserialize)]
        struct Example(#[serde(with = "serde_tensor")] Tensor);

        for _ in 0..100 {
            let orig = Example(Tensor::randn(
                &[3, 2, 4],
                (Kind::Float, Device::cuda_if_available()),
            ));
            let text = serde_json::to_string(&orig)?;
            let recovered = serde_json::from_str(&text)?;

            let Example(orig_tensor) = orig;
            let Example(recovered_tensor) = recovered;

            assert_eq!(orig_tensor.size(), recovered_tensor.size());
            assert_eq!(orig_tensor.kind(), recovered_tensor.kind());
            assert_eq!(orig_tensor, recovered_tensor);
        }

        for _ in 0..100 {
            let orig = Example(Tensor::randint(
                1024,
                &[3, 2, 4],
                (Kind::Float, Device::cuda_if_available()),
            ));
            let text = serde_json::to_string(&orig)?;
            let recovered = serde_json::from_str(&text)?;

            let Example(orig_tensor) = orig;
            let Example(recovered_tensor) = recovered;

            assert_eq!(orig_tensor.size(), recovered_tensor.size());
            assert_eq!(orig_tensor.kind(), recovered_tensor.kind());
            assert_eq!(orig_tensor, recovered_tensor);
        }

        Ok(())
    }
}
