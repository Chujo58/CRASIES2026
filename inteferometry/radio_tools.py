import numpy as np
import numba as nb
def itrs_to_enu(xyz, ref_lat, ref_lon):
    if len(xyz.shape)==1:
        xyz = xyz.reshape(-1,3)
    enu=np.zeros(xyz.shape,dtype=xyz.dtype)
    nn = xyz.shape[0]
    _lat=np.deg2rad(ref_lat)
    _lon=np.deg2rad(ref_lon)
    sin_lat = np.sin(_lat)
    cos_lat = np.cos(_lat)
    sin_lon = np.sin(_lon)
    cos_lon = np.cos(_lon)
    for i in range(nn):
        xyz_use = xyz[i]
        enu[i, 0] = -sin_lon * xyz_use[0] + cos_lon * xyz_use[1]
        enu[i, 1] = (
          - sin_lat * cos_lon * xyz_use[0]
          - sin_lat * sin_lon * xyz_use[1]
          + cos_lat * xyz_use[2]
        )
        enu[i, 2] = (
          cos_lat * cos_lon * xyz_use[0]
          + cos_lat * sin_lon * xyz_use[1]
          + sin_lat * xyz_use[2]
        )
    return enu

@nb.njit(parallel=True)
def geo_delay_from_enu(bls,az,alt):
    # all angles in rad
    npix = alt.shape[0]
    nbl = bls.shape[0] #each row is x, y, z
    delays = np.empty((nbl,npix),dtype='float64')
    c=299792458
    for pix in nb.prange(npix):
        cos_alt = np.cos(alt[pix])
        sin_alt = np.sin(alt[pix])
        cos_az = np.cos(az[pix])
        sin_az = np.sin(az[pix])
        #unit vectors of source in ENU
        u0 = sin_az * cos_alt #east
        u1 = cos_az * cos_alt #north
        u2 = sin_alt          #up
        for bl in range(nbl):
            ea,no,up = bls[bl]
            p = u0 * ea + u1 * no + u2 * up #dot product of source dir. and baseline
            delays[bl, pix] = p/c
    return delays

def hadec_to_azalt(ha, dec, lat):
    """
    Convert Hour Angle and Declination to Altitude and Azimuth.
    All inputs and outputs must be in radians.
    
    Parameters:
    -----------
    ha : float or array-like
        Hour Angle(s) in radians
    dec : float or array-like
        Declination(s) in radians
    lat : float
        Observer's latitude in radians
        
    Returns:
    --------
    az : float or array-like
        Azimuth angle(s)
    alt : float or array-like
        Altitude angle(s)
    """
    sinha = np.sin(ha)
    cosha = np.cos(ha)
    sindec = np.sin(dec)
    cosdec = np.cos(dec)
    sinlat = np.sin(lat)
    coslat = np.cos(lat)
    
    aa = sinlat * sindec + coslat * cosdec * cosha
    aa = np.clip(aa, -1.0, 1.0)
    alt = np.arcsin(aa)
    sinaz = -cosdec * sinha
    cosaz = (coslat * sindec - sinlat * cosdec * cosha)
    az = np.arctan2(sinaz, cosaz)
    az = (az + 2 * np.pi) % (2 * np.pi)
    return az, alt

def azalt_to_hadec(az, alt, lat):
    """
    Convert Altitude and Azimuth to Hour Angle and Declination.
    All inputs and outputs must be in radians.
    
    Parameters:
    -----------
    az : float or array-like
        Azimuth angle(s) in radians
    alt : float or array-like
        Altitude angle(s) in radians
    lat : float
        Observer's latitude in radians
        
    Returns:
    --------
    ha : float or array-like
        Hour Angle(s) in radians [-pi, pi]
    dec : float or array-like
        Declination(s) in radians [-pi/2, pi/2]
    """
    sinalt = np.sin(alt)
    cosalt = np.cos(alt)
    sinaz = np.sin(az)
    cosaz = np.cos(az)
    sinlat = np.sin(lat)
    coslat = np.cos(lat)

    sindec = sinalt * sinlat + cosalt * coslat * cosaz
    
    # Clip to strictly [-1, 1] to avoid NaN errors from floating point inaccuracies
    sindec = np.clip(sindec, -1.0, 1.0)
    dec = np.arcsin(sindec)

    y = -sinaz * cosalt
    x = sinalt * coslat - cosalt * sinlat * cosaz
    ha = np.arctan2(y, x)
    ha = ha % (2 * np.pi)
    return ha, dec