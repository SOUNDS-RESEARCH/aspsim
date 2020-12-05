import numpy as np
import scipy.spatial.distance as distfuncs
import scipy.special as special
import scipy.signal as sig
import quadpy as qp

from pathos.helpers import freeze_support

import ancsim.utilities as util
import ancsim.signal.filterdesign as fd
import ancsim.integration.montecarlo as mc
import ancsim.integration.pointgenerator as gen


# INTERPOLATE SOUND FIELD FROM SET OF POINTS TO SET OF POINTS
# def kernelHelmholtz3d(toPoints, fromPoints, waveNum):
#     distMat = distfuncs.cdist(fromPoints, toPoints)[None, :, :]
#     return special.spherical_jn(0, distMat * waveNum)

def kernelHelmholtz2d(toPoints, fromPoints, waveNum):
    distMat = distfuncs.cdist(fromPoints, toPoints)[None, :, :]
    return special.j0(distMat * waveNum)

# def kernelHelmholtz3d(toPoints, fromPoints, waveNum):
#     """toPoints is shape (numToPoints, 3)
#         fromPoints is shape (numFromPoints, 3)
#         waveNum is shape (numFreqs)

#         returns shape (numFreqs, numFromPoints, numToPoints)
#     """
#     distMat = distfuncs.cdist(fromPoints, toPoints)[None, :, :]
#     return special.spherical_jn(0, distMat * waveNum[:,None,None])

def kernelHelmholtz3d(points1, points2, waveNum):
    """points1 is shape (numPoints1, 3)
        points2 is shape (numPoints2, 3)
        waveNum is shape (numFreqs)

        returns shape (numFreqs, numPoints1, numPoints2)
    """
    distMat = distfuncs.cdist(points1, points2)
    return special.spherical_jn(0, distMat[None,:,:] * waveNum[:,None,None])

def kernelDirectional3d(points1, points2, waveNum, angle, beta):
    """points1 is shape (numPoints1, 3)
        points2 is shape (numPoints2, 3)
        waveNum is shape (numFreqs)
        angle is tuple (theta, phi) defined as in util.spherical2cart
        
        returns shape (numFreqs, numPoints1, numPoints2)
    """
    #distMat = distfuncs.cdist(fromPoints, toPoints)[None, :, :]
    rDiff = points1[:,None,:] - points2[None,:,:]

    angleFactor = beta * util.spherical2cart(np.ones((1,1)), np.array(angle)[None,:])[None,None,...]

    posFactor = 1j * waveNum[:,None,None,None] * rDiff[None,...]

    return special.spherical_jn(0, 1j*np.sum((angleFactor + posFactor)**2, axis=-1))
    
    # \kappa(r_1, r_2) = j_0( j (
    # (\beta \sin\theta\cos\phi + j k x_12)^2 + 
    # (\beta \sin\theta\sin\phi + j k y_12)^2 + 
    # (\beta\cos\theta + j k z_12)^2 )^1/2)



def integrableAwFunc(k, posErr, beta=0, ang=0):
    def intFunc(r):
        r_diff = (np.tile((r.T)[None,:,:], (posErr.shape[0],1,1)) - np.tile(posErr[:,None,:], (1,r.shape[1],1)))[None,:,:,:]
        distance = 1j*np.sqrt((beta*np.cos(ang) + 1j*k[:,None,None]*r_diff[:,:,:,0])**2 + (beta*np.sin(ang) + 1j*k[:,None,None]*r_diff[:,:,:,1])**2)
        kappa = special.jn(0, distance)
        funcVal = kappa[:, :, None, :].conj() * kappa[:, None, :, :]
        return funcVal
    return intFunc


def soundfieldInterpolationFIR(
    toPoints, fromPoints, irLen, regParam, numFreq, spatialDims, samplerate, c
):
    assert numFreq > irLen

    freqFilter = soundfieldInterpolation(
        toPoints, fromPoints, numFreq, regParam, spatialDims, samplerate, c
    )

    kiFilter = fd.firFromFreqsWindow(freqFilter, irLen)
    return kiFilter


def soundfieldInterpolation(
    toPoints, fromPoints, numFreq, regParam, spatialDims, samplerate, c
):
    """Calculates the vector or matrix used to interpolate from fromPoints to toPoints"""
    if spatialDims == 3:
        kernelFunc = kernelHelmholtz3d
    elif spatialDims == 2:
        kernelFunc = kernelHelmholtz2d
    else:
        raise ValueError

    assert numFreq % 2 == 0

    freqs = fd.getFrequencyValues(numFreq, samplerate)#[:, None, None]
    waveNum = 2 * np.pi * freqs / c
    ipParams = getKRRParameters(kernelFunc, regParam, toPoints, fromPoints, waveNum)
    ipParams = fd.insertNegativeFrequencies(ipParams, even=True)
    return ipParams


def getKRRParameters(kernelFunc, regParam, outputArg, dataArg, *args):
    """Calculates parameter vector or matrix given a kernel function for Kernel Ridge Regression.
    Both dataArg and outputArg should be formatted as (numPoints, pointDimension)
    kernelFunc should return args as (numFreq, numDataPoints, numOutPoints)
    returns params of shape (numFreq, outputPoints, inputPoints)"""
    dataDim = dataArg.shape[0]
    K = kernelFunc(dataArg, dataArg, *args)
    Kreg = K + regParam * np.eye(dataDim)
    kappa = kernelFunc(outputArg, dataArg, *args)

    params = np.transpose(np.linalg.solve(Kreg, kappa), (0, 2, 1))
    return params


# ==================================================================================


# FREQUENCY DOMAIN 2D DISC KERNEL INTERPOLATION WEIGHTING FILTER
def kernelInterpolationFR(errorMicPos, freq, regParam, truncOrder, radius, c):
    if isinstance(freq, (int, float)):
        freq = np.array([freq])
    if len(freq.shape) == 1:
        freq = freq[:, np.newaxis, np.newaxis]
    waveNumber = 2 * np.pi * freq / c
    K = getK(errorMicPos, waveNumber)
    P = getP(regParam, K)
    S = getS(truncOrder, waveNumber, errorMicPos)
    Gamma = getGamma(truncOrder, waveNumber, radius)
    A = (
        np.transpose(P.conj(), (0, 2, 1))
        @ np.transpose(S.conj(), (0, 2, 1))
        @ Gamma
        @ S
        @ P
    )
    return A


def getK(pos, k):
    distanceMat = distfuncs.cdist(pos, pos)
    K = special.j0(k * distanceMat)
    return K


def getP(regParam, K):
    return np.linalg.pinv(K + regParam * np.eye(K.shape[-1]))


def getGamma(maxOrder, k, R):
    matLen = 2 * maxOrder + 1
    diagValues = smallGamma(np.arange(-maxOrder, maxOrder + 1), k, R)
    gamma = np.zeros((diagValues.shape[0], matLen, matLen))

    gamma[:, np.arange(matLen), np.arange(matLen)] = diagValues
    return gamma


def smallGamma(mu, k, R):
    Jfunc = special.jv((mu - 1, mu, mu + 1), k * R)
    return np.pi * (R ** 2) * ((Jfunc[:, 1, :] ** 2) - Jfunc[:, 0, :] * Jfunc[:, 2, :])


def getS(maxOrder, k, positions):
    r, theta = util.cart2pol(positions[:, 0], positions[:, 1])

    mu = np.arange(-maxOrder, maxOrder + 1)[:, np.newaxis]
    S = special.jv(mu, k * r) * np.exp(theta * mu * (-1j))
    return S


# ======================================================================================
# Functions for kernel interpolation in 2d
# Equals routines for C2 in earlier versions
# C2 is integrating directly on INT(B_m1(k,r)B_m2(t,r))


def getCMatrixDisc2d(errorPos, integrationOrder, radius, numFreqSamples, samplerate, c):
    intFunc = getIntegrableFunc2d(errorPos, numFreqSamples, samplerate, c)
    intScheme = qp.disk.lether(integrationOrder)
    c = intScheme.integrate(intFunc, 0, radius)
    return c


def getIntegrableFunc2d(errorPos, filtLen, regParam, numFreqSamples, samplerate, c):
    numError = errorPos.shape[0]
    freqs = ((samplerate / (2 * numFreqSamples)) * np.arange(numFreqSamples))[
        :, None, None
    ]
    waveNum = freqs * 2 * np.pi / c

    distMat = distfuncs.cdist(errorPos, errorPos)[None, :, :]
    K = special.j0(distMat * waveNum)
    Kinv = np.linalg.inv(K + regParam * np.eye(numError))

    def B(r):
        distance = np.transpose(
            distfuncs.cdist(np.transpose(r, (1, 0)), errorPos), (1, 0)
        )[None, :, :]
        kappa = special.j0(distance * waveNum)
        freqFunc = Kinv @ kappa
        ir = fd.tdFilterFromFreq(
            np.transpose(freqFunc, (1, 2, 0)),
            filtLen,
            method="window",
            window="hamming",
        )
        return np.transpose(ir, (2, 0, 1))

    def funcProduct(r):
        b = B(r)
        tot = b[None, None, :, :, :] * b[:, :, None, None, :]
        return tot

    return funcProduct


# ====================================================================================
# Functions for kernel interpolation in 3d
# Equals routines for c2 in earlier versions
# def getCMatrixCylinder3d_old(errorPos, integrationSamples=2000, radius=s.TARGET_RADIUS, height=s.TARGET_HEIGHT):
#     f = getIntegrableFunc3d(errorPos)
#     val = mcIntegrateCylinder(f, radius, height, integrationSamples)
#     return val


def getCMatrixCylinder3d(errorPos, integrationSamples, radius, height):
    f = getIntegrableFunc3d(errorPos)

    def pointGenerator(numSamples):
        r = np.sqrt(np.random.rand(numSamples)) * radius
        angle = np.random.rand(numSamples) * 2 * np.pi
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        z = np.random.uniform(-height / 2, height / 2, size=numSamples)
        points = np.stack((x, y, z))
        return points

    totVol = radius ** 2 * np.pi * height
    kiFilter = mc.integrateMp(f, pointGenerator, integrationSamples, totVol)
    return kiFilter


def getCMatrixCylinder3d_seq(errorPos, integrationSamples, radius, height):
    f = getIntegrableFunc3d(errorPos)

    def pointGenerator(numSamples):
        r = np.sqrt(np.random.rand(numSamples)) * radius
        angle = np.random.rand(numSamples) * 2 * np.pi
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        z = np.random.uniform(-height / 2, height / 2, size=numSamples)
        points = np.stack((x, y, z))
        return points

    totVol = radius ** 2 * np.pi * height
    kiFilter = mc.integrate(f, pointGenerator, integrationSamples, totVol)
    return kiFilter


def getCMatrixBlock3d(
    errorPos, integrationSamples, dims, height, numFreqSamples, samplerate, c
):
    f = getIntegrableFunc3d(errorPos, numFreqSamples, samplerate, c)

    def pointGenerator(numSamples):
        x = np.random.uniform(-dims[0] / 2, dims[0] / 2, numSamples)
        y = np.random.uniform(-dims[1] / 2, dims[1] / 2, numSamples)
        z = np.random.uniform(-height / 2, height / 2, numSamples)
        points = np.stack((x, y, z))
        return points

    volume = dims[0] * dims[1] * height
    val = mc.integrateMp(f, pointGenerator, integrationSamples, volume)
    return val


def getIntegrableFunc3d(errorPos, filtLen, regParam, numFreqSamples, samplerate, c):
    numError = errorPos.shape[0]
    freqs = ((samplerate / (2 * numFreqSamples)) * np.arange(numFreqSamples))[
        :, None, None
    ]
    waveNum = freqs * 2 * np.pi / c

    distMat = distfuncs.cdist(errorPos, errorPos)[None, :, :]
    K = special.spherical_jn(0, distMat * waveNum)
    Kinv = np.linalg.inv(K + regParam * np.eye(numError))

    def B(r):
        distance = np.transpose(
            distfuncs.cdist(np.transpose(r, (1, 0)), errorPos), (1, 0)
        )[None, :, :]
        kappa = special.spherical_jn(0, distance * waveNum)
        freqFunc = Kinv @ kappa
        ir = fd.tdFilterFromFreq(
            np.transpose(freqFunc, (1, 2, 0)),
            filtLen,
            method="window",
            window="hamming",
        )
        return np.transpose(ir, (2, 0, 1))

    def funcProduct(r):
        b = B(r)
        tot = b[None, None, :, :, :] * b[:, :, None, None, :]
        return tot

    return funcProduct


# ======================================================================================
# Corresponds to #R_{m_1, m_2}(u,v) in derivations
def getRtimedomainCylinder3d(errorPos, blockSize, samplerate, c):
    R = getRMatrixCylinder3d(errorPos, samplerate, c)
    tdR = np.fft.ifft(R, axis=0)
    tdR = np.fft.ifft(tdR, axis=2)
    tdR = np.concatenate(
        (tdR[-blockSize + 1 :, :, :, :], tdR[0 : blockSize + 1, :, :, :]), axis=0
    )
    tdR = np.concatenate(
        (tdR[:, :, -blockSize + 1 :, :], tdR[:, :, 0 : blockSize + 1, :]), axis=2
    )
    tdR = sig.windows.hamming(blockSize * 2)[:, None, None, None] * tdR
    tdR = sig.windows.hamming(blockSize * 2)[None, None, :, None] * tdR
    return tdR


def getRMatrixCylinder3d(
    errorPos, numFreq, numMCSamples, radius, height, regParam, samplerate, c
):
    numError = errorPos.shape[0]
    freqs = ((samplerate / (2 * numFreq)) * np.arange(numFreq + 1))[:, None, None]
    waveNum = freqs * 2 * np.pi / c
    distMat = distfuncs.cdist(errorPos, errorPos)[None, :, :]
    K = special.spherical_jn(0, distMat * waveNum)
    Kinv = np.linalg.inv(K + regParam * np.eye(numError))

    func = getIntegrableFreqFunc3d(waveNum, errorPos)
    pointGen, vol = gen.cylinder(radius, height)
    integralVal = mc.integrate(func, pointGen, numMCSamples, vol, numPerIter=5)

    integralVal = np.sum(
        Kinv[:, :, :, None, None] * integralVal[:, :, None, :, :], axis=1
    )
    integralVal = np.sum(
        Kinv[None, None, :, :, :] * integralVal[:, :, :, :, None], axis=3
    )

    allFreqs = np.concatenate(
        (integralVal, np.flip(integralVal[1:-1, :, :, :], axis=0)), axis=0
    )
    allFreqs = np.concatenate(
        (allFreqs, np.flip(allFreqs[:, :, 1:-1, :], axis=2)), axis=2
    )
    return allFreqs


def getIntegrableFreqFunc3d(waveNum, errorPos):
    def intFunc(r):
        distance = np.transpose(
            distfuncs.cdist(np.transpose(r, (1, 0)), errorPos), (1, 0)
        )[None, :, :]
        kappa = special.spherical_jn(0, distance * waveNum)
        funcVal = kappa[None, None, :, :, :] * kappa[:, :, None, None, :]
        return funcVal

    return intFunc


# ITO DERIVATION ======================================================================================
# Corresponds to Ito-sans derivation. the A matrix

# def getAMatrixCylinder3d(errorPos, numFreq, numMCSamples, radius, height,
#                                 regParam, samplerate = s.SAMPLERATE):
#     numError = errorPos.shape[0]
#     freqs = ((samplerate / (2*numFreq)) * np.arange(numFreq+1))[:,None,None]
#     waveNum = freqs * 2*np.pi/s.C

#     func = integrableAFunc(waveNum, errorPos)
#     pointGen, vol = gen.cylinder(radius, height)
#     integralVal = mc.integrate(func, pointGen, numMCSamples, vol)

#     distMat = distfuncs.cdist(errorPos, errorPos)[None,:,:]
#     K = special.spherical_jn(0,distMat*waveNum)
#     Kinv = np.linalg.inv(K + regParam * np.eye(numError))
#     A = Kinv @ integralVal @ Kinv
#     A = np.concatenate((A, np.flip(A[1:-1,:,:], axis=0)), axis=0)
#     return A


def integrableAFunc(waveNum, errorPos):
    def intFunc(r):
        distance = np.transpose(
            distfuncs.cdist(np.transpose(r, (1, 0)), errorPos), (1, 0)
        )[None, :, :]
        kappa = special.spherical_jn(0, distance * waveNum)
        funcVal = kappa[:, :, None, :] * kappa[:, None, :, :]
        return funcVal

    return intFunc


# pointGen, volume = gen.selectPointGen(config, randomState)
def getAKernelFreqDomain3d(
    errorPos, numFreq, kernelReg, mcPointGen, mcVolume, mcNumPoints, samplerate, c
):
    """Filter length will be 2*numFreq. The parameter sets the number of positive frequency bins"""
    numError = errorPos.shape[0]
    freqs = ((samplerate / (2 * numFreq)) * np.arange(numFreq + 1))[:, None, None]
    waveNum = freqs * 2 * np.pi / c

    func = integrableAFunc(waveNum, errorPos)

    integralVal = mc.integrate(func, mcPointGen, mcNumPoints, mcVolume)

    distMat = distfuncs.cdist(errorPos, errorPos)[None, :, :]
    K = special.spherical_jn(0, distMat * waveNum)
    Kinv = np.linalg.inv(K + kernelReg * np.eye(numError))
    A = Kinv @ integralVal @ Kinv
    A = np.concatenate((A, np.flip(A[1:-1, :, :], axis=0)), axis=0)
    return A


def getAKernelTimeDomain3d(
    errorPos,
    filtLen,
    kernelReg,
    mcPointGen,
    mcVolume,
    mcNumPoints,
    numFreq,
    samplerate,
    c,
):
    # assert(numFreq >= s.FILTLENGTH) # Dont remember why this assertion exists
    assert (
        numFreq >= filtLen
    )  # You need more samples in the frequency domain before truncating
    assert filtLen % 2 == 1  # With odd number of taps, you get an integer group delay.

    A = getAKernelFreqDomain3d(
        errorPos, numFreq, kernelReg, mcPointGen, mcVolume, mcNumPoints, samplerate, c
    )
    halfLen = int(filtLen / 2)
    tdA = np.real(np.fft.ifft(A, axis=0))
    tdA = np.concatenate((tdA[-halfLen:, :, :], tdA[0 : halfLen + 1, :, :]), axis=0)
    tdA = sig.windows.hamming(filtLen)[:, None, None] * tdA
    tdA = np.transpose(tdA, (1, 2, 0))
    assert tdA.shape[-1] == filtLen
    return tdA


def tdAFromFreq(A, filtLen):
    assert filtLen % 2 == 1

    halfLen = filtLen // 2

    tdA = np.real(np.fft.ifft(A, axis=0))
    tdA = np.concatenate((tdA[-halfLen:, :, :], tdA[0 : halfLen + 1, :, :]), axis=0)
    tdA = sig.windows.hamming(filtLen)[:, None, None] * tdA
    tdA = np.transpose(tdA, (1, 2, 0))
    return tdA


# ====================================================================================

# Filter length will be double numFreq. The parameters sets the amount of positive frequency bins
# def getAMatrixRect3d(errorPos, numFreq, numMCSamples, dim, height,
#                  regParam, samplerate = s.SAMPLERATE):
#     numError = errorPos.shape[0]
#     freqs = ((samplerate / (2*numFreq)) * np.arange(numFreq+1))[:,None,None]
#     waveNum = freqs * 2*np.pi/s.C

#     func = integrableAFunc(waveNum, errorPos)
#     pointGen, vol = gen.block([dim, dim], height)
#     integralVal = mc.integrate(func, pointGen, numMCSamples, vol)

#     distMat = distfuncs.cdist(errorPos, errorPos)[None,:,:]
#     K = special.spherical_jn(0,distMat*waveNum)
#     Kinv = np.linalg.inv(K + regParam * np.eye(numError))
#     A = Kinv @ integralVal @ Kinv
#     A = np.concatenate((A, np.flip(A[1:-1,:,:], axis=0)), axis=0)
#     return A


# def getTimeDomainACylinder3d_odd(errorPos, filtLen, numMCSamples, radius, height,
#                                 numFreq, regParam, samplerate = s.SAMPLERATE):
#     #numFreq default = 1024
#     assert(filtLen % 2 == 1)
#     assert(numFreq >= s.FILTLENGTH)
#     halfLen = int(filtLen / 2)
#     A = getAMatrixCylinder3d(errorPos, numMCSamples=numMCSamples,
#                             radius=radius, height=height,
#                             numFreq=numFreq, regParam=regParam,
#                             samplerate=samplerate)

#     tdA = np.real(np.fft.ifft(A, axis=0))
#     tdA = np.concatenate((tdA[-halfLen+1:,:,:], tdA[0:halfLen,:,:]), axis=0)
#     tdA = sig.windows.hamming(halfLen*2-1)[:,None,None] * tdA

#     return tdA


def getTimeDomainARect3d_even(
    errorPos, filtLen, numMCSamples, targetDim, height, numFreq, regParam, samplerate
):
    """TAKE CARE! Gives non-integer group delay. When delaying to get a linear phase/zero phase filter,
    it will need interpolation."""

    # numfreq default = 1024
    assert filtLen % 2 == 0
    # assert(numFreq >= s.FILTLENGTH)
    halfLen = filtLen // 2
    A = getAMatrixRect3d(
        errorPos,
        numMCSamples=numMCSamples,
        dim=targetDim,
        height=height,
        numFreq=numFreq,
        regParam=regParam,
        samplerate=samplerate,
    )

    tdA = np.real(np.fft.ifft(A, axis=0))
    tdA = np.concatenate(
        (tdA[-halfLen + 1 :, :, :], tdA[0 : halfLen + 1, :, :]), axis=0
    )
    tdA = sig.windows.hamming(halfLen * 2)[:, None, None] * tdA

    return tdA


# def getTimeDomainARect3d_odd(errorPos, filtLen, numMCSamples, targetDim, height,
#                             numFreq, regParam, samplerate = s.SAMPLERATE):
#     #numfreq default = 1024
#     assert(numFreq >= s.FILTLENGTH)
#     assert(filtLen % 2 == 1)
#     A = getAMatrixRect3d(errorPos, numMCSamples=numMCSamples,
#                                 dim=targetDim, height=height,
#                                 numFreq=numFreq, regParam=regParam,
#                                 samplerate=samplerate)

#     halfLen = int(filtLen / 2)
#     tdA = np.real(np.fft.ifft(A, axis=0))
#     tdA = np.concatenate((tdA[-halfLen:,:,:], tdA[0:halfLen+1,:,:]), axis=0)
#     tdA = sig.windows.hamming(filtLen)[:,None,None] * tdA
#     assert(tdA.shape[0] == filtLen)
#     return tdA


# ========================================================================
# DERIVATION 12


def getKernel12TimeDomain3d(
    errorPos, filtLen, config, kernelReg, numFreq, samplerate, c
):
    func = kernel12TimeDomainIntegrableFunc(
        errorPos, filtLen, numFreq, kernelReg, samplerate, c
    )
    pointGen, volume = gen.selectPointGen(config)
    integralVal = mc.integrate(func, pointGen, config["MCPOINTS"], volume)
    return integralVal


def kernel12TimeDomainIntegrableFunc(
    errorPos, filtLen, numFreq, kernelreg, samplerate, c
):
    freqs = ((samplerate / (2 * numFreq)) * np.arange(numFreq + 1))[:, None, None]
    waveNum = freqs * 2 * np.pi / c

    numError = errorPos.shape[0]
    distMat = distfuncs.cdist(errorPos, errorPos)[None, :, :]
    K = special.spherical_jn(0, distMat * waveNum)
    Kinv = np.linalg.inv(K + kernelreg * np.eye(numError))

    preSumLen = 2 * filtLen + 1
    assert preSumLen % 2 == 1
    halfPreSumLen = int(preSumLen / 2)

    def kernelFilter(r):
        distance = np.transpose(
            distfuncs.cdist(np.transpose(r, (1, 0)), errorPos), (1, 0)
        )[None, :, :]
        kappa = special.spherical_jn(0, distance * waveNum)
        kappa = np.transpose(kappa, (2, 0, 1))[:, :, :, None]

        fdFilter = Kinv[None, :, :, :] @ kappa
        fdFilter = np.concatenate(
            (fdFilter, np.flip(fdFilter[:, 1:-1, :, :], axis=1)), axis=1
        )

        tdFilter = np.real(np.fft.ifft(fdFilter, axis=1))
        tdFilter = np.concatenate(
            (
                tdFilter[:, -halfPreSumLen:, :, :],
                tdFilter[:, 0 : halfPreSumLen + 1, :, :],
            ),
            axis=1,
        )
        # tdFilter = np.transpose(tdFilter, (0,1,2))

        totFiltLen = 2 * preSumLen - 1
        totOutputFilt = np.zeros((r.shape[-1], numError, numError, totFiltLen))
        for i in range(preSumLen):
            tempFilt = tdFilter[:, i : i + 1, :, :] * np.transpose(
                tdFilter, (0, 1, 3, 2)
            )
            totOutputFilt[:, :, :, i : i + preSumLen] += np.transpose(
                tempFilt, (0, 2, 3, 1)
            )

        midPoint = totFiltLen // 2
        halfFiltLen = filtLen // 2
        outputFilt = totOutputFilt[
            :, :, :, midPoint - halfFiltLen : midPoint + halfFiltLen + 1
        ]
        outputFilt *= sig.windows.hamming(filtLen)[None, None, None, :]
        outputFilt = np.transpose(outputFilt, (1, 2, 3, 0))
        return outputFilt

    return kernelFilter


# =====================================================================================
def findNecessaryKernLen(errorPos, radius, height):
    maxLen = 415
    allowedError = 0.1
    numError = errorPos.shape[0]

    f = getIntegrableFunc3d(errorPos, filtLen=maxLen)
    pointGen, vol = gen.cylinder(radius, height)

    C = mc.integrate(f, pointGen, 1000, vol)
    reqFiltLen = np.zeros((2, maxLen, numError, numError))

    midIdx = int(maxLen / 2)
    for eSameAxis in range(numError):
        for eOtherAxis in range(numError):
            for otherFiltAxis in range(maxLen):
                currentFilt = C[otherFiltAxis, eOtherAxis, :, eSameAxis]
                currentFilt *= 1 / np.abs(currentFilt).max()
                # energy = np.sum(currentFilt**2)
                # if currentFilt.shape % 2 == 0:
                energyNeeded = np.sum(currentFilt[midIdx:] ** 2) * (1 - allowedError)
                # else:

                energyCount = 0
                for i in range(midIdx, maxLen):
                    energyCount += C[otherFiltAxis, eOtherAxis, i, eSameAxis] ** 2
                    if energyCount > energyNeeded:
                        reqFiltLen[0, otherFiltAxis, eSameAxis, eOtherAxis] = (
                            i + 1 - midIdx
                        ) * 2
                        break
    for eSameAxis in range(numError):
        for eOtherAxis in range(numError):
            for otherFiltAxis in range(maxLen):
                currentFilt = C[:, eSameAxis, otherFiltAxis, eOtherAxis]
                currentFilt *= 1 / np.abs(currentFilt).max()
                energyNeeded = np.sum(currentFilt[midIdx:] ** 2) * (1 - allowedError)
                energyCount = 0
                for i in range(midIdx, maxLen):
                    energyCount += C[i, eSameAxis, otherFiltAxis, eOtherAxis] ** 2
                    if energyCount > energyNeeded:
                        reqFiltLen[1, otherFiltAxis, eSameAxis, eOtherAxis] = (
                            i + 1 - midIdx
                        ) * 2
                        break
    if maxLen % 2 == 1:
        reqFiltLen -= 1
    printReqFiltLen(reqFiltLen, allowedError, maxLen)
    return reqFiltLen


def printReqFiltLen(reqFiltLen, tol, maxLen):
    print("Tolerance: ", tol)
    print("Max length of IR which was tested: ", maxLen)

    print("Required Filter Length Set 1: ", reqFiltLen[0, :, :, :].max())
    print("Required Filter Length Set 2: ", reqFiltLen[1, :, :, :].max())


# ====================================================================================


def test_c1_c2_similarity(pos):
    np.random.seed(1)
    c1 = getCMatrixCylinder3d(pos["error"], 200)

    dif = np.abs(c1 - c2)
    dif = np.max(dif, axis=0)
    dif = dif.reshape((-1, dif.shape[1]))
    plt.plot(dif.T)
    plt.show()


def testSpeedKernelFilt(errorPos):
    start = time.time()
    getCMatrixCylinder3d_mp(errorPos, 1000)
    mp = time.time() - start
    start = time.time()
    getCMatrixCylinder3d_new(errorPos, 1000)
    normal = time.time() - start
    print("MP: ", mp)
    print("Normal: ", normal)
