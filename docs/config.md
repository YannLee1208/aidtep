# Configuration

The `config.yaml` file is used to configure the AIDTEP platform. Below is an example configuration file and explanations of each section.


## Models

The AIDTEP platform supports various models. Below are the supported models and their configurations.


### NVT-ResNet

NVT-ResNet is based on ResNet, a type of deep neural network model known for its residual connections. To use ResNet, configure your `config.yaml` as follows:

```yaml
model:
  type: NVT_ResNet
```

## Interpolation Methods

The AIDTEP platform supports various interpolation methods. Below are the supported interpolation methods and their configurations.

## Supported Interpolation Methods

### Voronoi Interpolation

Voronoi interpolation is a method that divides the space into regions based on the distance to a specific set of points. To use Voronoi interpolation, configure your `config.yaml` as follows:

```yaml
interpolation:
  method: voronoi
```
### Voronoi Linear Interpolation

Voronoi linear interpolation is a method that divides the space into regions based on the distance to a specific set of points and then interpolates the values linearly. To use Voronoi linear interpolation, configure your `config.yaml` as follows:

```yaml
interpolation:
  method: voronoi_linear
```
