# Numpy 转Tensor

#### 通过tf.convert_to_tensor(数据名,dtype=数据类型(可选))

```java
import tensorflow as tf
import numpy as np
a = np.arange(0,5)
t = tf.convert_to_tensor(a,dtype=tf.int64)
print(a)
print(t)
```

