import numpy as np
from phi import struct
from phi.math.math_util import is_static_shape

# creates normal distributed noise that can vary over the batch
def generateNoise(grid, var, mean=0, seed=0, dtype=np.float32):
    size = grid.data.shape
    rand = np.random.RandomState(seed)
    def array(shape):
        result = np.ones(size)
        for i in range(size[0]):
            result[i] = rand.normal(mean, var[i], size=size[1:]).astype(dtype)
        return result
    return struct.map(array, grid, leaf_condition=is_static_shape)

# creates randomized parameters for grid generation (v_parameter in paramType is varied over batch)
def generateParams(paramType, batch, dim, noiseStrength, vf1, vf2, vf3, vf4, vf5, vf7, vo1, vo2, vod, vnoise):
    f1 = (-0.5 + np.random.rand(dim,dim) ) * 0.4 # range +- 0.4 , initial main dir
    f2 = (-0.5 + np.random.rand(dim,dim) ) * 0.4 # reduced ranger for frequencies , freq1
    f3 = (-0.5 + np.random.rand(dim,dim) ) * 0.3 # freq2
    f4 = (-0.5 + np.random.rand(dim,dim) ) * 0.3 # freq3
    f5 = (-0.5 + np.random.rand(dim,dim) ) * 0.2 # freq4
    f7 = (-0.5 + np.random.rand(dim,dim) ) * 100.0 # forcing shift, dir&speed
    o1 = 0. + ( np.random.rand(dim,dim)*100. ) # offsets, minor influence
    o2 = 0. + ( np.random.rand(dim,dim)*100. ) #
    nu = 0.0002 * ( 1+ np.random.rand()*500. ) # diffusion

    # switch between "static" (near 1) and "forced" (=0) cases smoothly
    sfCase = 0. + ( np.random.rand(dim,dim)*1. )
    # enlarge regular init for "static"
    staticF = (1.+2.*sfCase)
    f1 *= staticF
    f2 *= staticF
    f3 *= staticF
    f4 *= staticF
    f5 *= staticF
    f6 = (1.-sfCase) * 0.1 # note, factor is just forcing strength scaling for f6 factor

    fd = 1. / (1. + np.random.randint(6, size=(dim,dim)))
    od = 0. +  np.random.rand(dim,dim)*100. 

    f1 = np.repeat(f1[np.newaxis,...], batch, axis=0)
    f2 = np.repeat(f2[np.newaxis,...], batch, axis=0)
    f3 = np.repeat(f3[np.newaxis,...], batch, axis=0)
    f4 = np.repeat(f4[np.newaxis,...], batch, axis=0)
    f5 = np.repeat(f5[np.newaxis,...], batch, axis=0)
    f6 = np.repeat(f6[np.newaxis,...], batch, axis=0)
    f7 = np.repeat(f7[np.newaxis,...], batch, axis=0)
    o1 = np.repeat(o1[np.newaxis,...], batch, axis=0)
    o2 = np.repeat(o2[np.newaxis,...], batch, axis=0)
    fd = np.repeat(fd[np.newaxis,...], batch, axis=0)
    od = np.repeat(od[np.newaxis,...], batch, axis=0)

    noise = noiseStrength * np.ones(batch) # normally constant noise level

    amount = np.repeat(np.arange(batch)[...,np.newaxis], dim, axis=1)
    amount = np.repeat(amount[...,np.newaxis], dim, axis=2)

    if paramType == "f1":        f1 += vf1 * amount
    elif paramType == "f1neg":   f1 -= vf1 * amount
    elif paramType == "f2":      f2 += vf2 * amount
    elif paramType == "f2neg":   f2 -= vf2 * amount
    elif paramType == "f3":      f3 += vf3 * amount
    elif paramType == "f3neg":   f3 -= vf3 * amount
    elif paramType == "f4":      f4 += vf4 * amount
    elif paramType == "f4neg":   f4 -= vf4 * amount
    elif paramType == "f5":      f5 += vf5 * amount
    elif paramType == "f5neg":   f5 -= vf5 * amount
    elif paramType == "f7":      f7 += vf7 * amount
    elif paramType == "f7neg":   f7 -= vf7 * amount
    elif paramType == "o1":      o1 += vo1 * amount
    elif paramType == "o1neg":   o1 -= vo1 * amount
    elif paramType == "o2":      o2 += vo2 * amount
    elif paramType == "o2neg":   o2 -= vo2 * amount
    elif paramType == "od":      od += vod * amount
    elif paramType == "odneg":   od -= vod * amount
    elif paramType == "noise":   noise = vnoise * np.arange(batch) # increasing noise level
    else: raise ValueError("Unknown parameter type!")

    params = {
        "nu" : np.array(nu),
        "f1" : f1,
        "f2" : f2,
        "f3" : f3,
        "f4" : f4,
        "f5" : f5,
        "f6" : f6,
        "f7" : f7,
        "o1" : o1,
        "o2" : o2,
        "fd" : fd,
        "od" : od,
        "noise" : noise,
    }
    return params


# utilizes parameters generated with generateParams for grid initialization
def createParameterizedGrid(grid, gridType, age, dim, params):
    p = params
    size = np.array(grid.shape)

    def array(shape):
        mesh = np.meshgrid(np.arange(0, size[1]), np.arange(0, size[2]), np.arange(0, size[3]), indexing='ij')
        mesh = np.transpose(np.asarray(mesh), (1,2,3,0))[...,np.newaxis]
        mesh = np.repeat(mesh, dim, axis=dim+1)
        result = np.zeros(size, dtype=np.float32)

        # vary grid over batches:
        for i in range(size[0]):
            if gridType == "vectorForcing":
                temp = age / size[1:4]
                timeOff = 0.5 * p["f7"][i] + p["f7"][i] * np.sin(temp * 3.0)
                d = (mesh + timeOff) / size[1:4]
                acc  = p["f2"][i] * np.sin(d * 2  * np.pi)
                acc += p["f3"][i] * np.sin(d * 4  * np.pi +     p["o1"][i])
                acc += p["f4"][i] * np.sin(d * 8  * np.pi +     p["o2"][i])
                acc += p["f5"][i] * np.sin(d * 16 * np.pi + 0.7*p["o1"][i])
                acc *= p["f6"][i]
                result[i] = np.sum(acc, axis=dim)

            elif gridType == "vectorComplex":
                d = mesh / size[1:4]
                acc  = p["f1"][i] + np.zeros_like(d)
                acc += p["f2"][i] * np.sin(d * 2  * np.pi +     p["o1"][i])
                acc += p["f3"][i] * np.sin(d * 4  * np.pi +     p["o2"][i])
                acc += p["f4"][i] * np.sin(d * 8  * np.pi + 0.4*p["o1"][i])
                acc += p["f5"][i] * np.sin(d * 16 * np.pi + 0.3*p["o2"][i])
                result[i] = np.sum(acc, axis=dim)

            elif gridType == "scalarComplex":
                d = mesh / size[1:4]
                acc  = p["f1"][i] + np.zeros_like(d)
                acc += p["f2"][i] * np.sin(d * 2  * np.pi +     p["o1"][i])
                acc += p["f3"][i] * np.sin(d * 4  * np.pi +     p["o2"][i])
                acc += p["f4"][i] * np.sin(d * 8  * np.pi + 0.4*p["o1"][i])
                acc += p["f5"][i] * np.sin(d * 16 * np.pi + 0.3*p["o2"][i])
                temp = np.sum(acc, axis=dim+1)
                result[i] = np.sum(temp, axis=dim, keepdims=True)
                
            elif gridType == "scalarSimple":
                d = mesh / size[1:4]
                acc = np.sin(p["fd"][i] * d * 24 * np.pi + p["od"][i])
                temp = np.sum(acc, axis=dim+1)
                result[i] = np.sum(temp, axis=dim, keepdims=True)
                result[i] *= result[i] * result[i] # cubed, sharper transitions
            else:
                raise ValueError("Unknown grid type")
        return result
    return struct.map(array, grid, leaf_condition=is_static_shape)


# helper function for mp4 export
def prepareRender(data, pad):
    lines = [ [], [] ]
    for i in range(data.shape[0]):
        for j in range(len(lines)):
            part = data[i]
            #part = np.flip(part, axis=0)
            if j == 0:
                part = np.mean(part, axis=0)
            else:
                part = part[int(data.shape[1]/2)]
            part = part - np.min(part)
            part = part / np.max(part)
            part = 255*part
            part = np.pad(part, ((pad,pad), (pad,pad), (0,0)) )
            lines[j].append(part.astype(np.uint8))

    for j in range(len(lines)):
        lines[j] = np.concatenate(lines[j], axis=1)

    return np.concatenate(lines, axis=0)