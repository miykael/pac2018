Classifier       : GNB
Postfix          : _scanner1
Chunks           : 5
Sphere Radius    : 3
N-th Element     : 10
Wall Time        : 00:11:28
Samples          : 464
Features         : 380789
Volume Dimension : (121, 145, 121)
Voxel  Dimension : (1.5, 1.5)
CPU              : 8

Chance Level     : 0.5
Accuracy (mean)  : 0.51576
Accuracy (std)   : 0.02378

%Sphere > +2STD  : 7.342%
%Sphere > +3STD  : 0.327%
%Sphere > +4STD  : 0.001%
vSphere > +2STD  : 27957
vSphere > +3STD  :  1247
vSphere > +4STD  :     4


Dataset Summary:
****************
Dataset: 464x380789@float64, <sa: chunks,targets>, <fa: voxel_indices>, <a: imgaffine,imghdr,imgtype,mapper,voxel_dim,voxel_eldim>
stats: mean=3.25946e-17 std=1 var=1 min=-6.26529 max=9.36192

Counts of targets in each chunk:
  chunks\targets 1.0 2.0
                 --- ---
       0.0        47  47
       1.0        47  47
       2.0        46  46
       3.0        46  46
       4.0        46  46

Summary for targets across chunks
  targets mean  std min max #chunks
    1     46.4 0.49  46  47    5
    2     46.4 0.49  46  47    5

Summary for chunks across targets
  chunks mean std min max #targets
    0     47   0   47  47     2
    1     47   0   47  47     2
    2     46   0   46  46     2
    3     46   0   46  46     2
    4     46   0   46  46     2
Sequence statistics for 464 entries from set [1.0, 2.0]
Counter-balance table for orders up to 2:
Targets/Order  O1      |   O2      |
     1.0:     231  1   |  230  2   |
     2.0:      0  231  |   0  230  |
Correlations: min=-1 max=0.99 mean=-0.0022 sum(abs)=2.3e+02<bound method Dataset.summary of Dataset(array([[ 0.32187755,  0.22005144,  2.36513688, ...,  1.40724836,
         1.67226456, -0.65486989],
       [-0.26247526, -0.56794228, -1.12228064, ...,  0.66325879,
         0.27032147,  0.74141088],
       [ 0.25694946,  0.35138372,  0.68370343, ..., -0.13033009,
        -0.60589296, -0.41203845],
       ...,
       [ 2.7617299 , -0.41338703, -1.02997857, ..., -0.09888623,
        -0.52292819, -1.29359795],
       [ 1.10155363,  0.37893811, -1.18293745, ...,  1.09245745,
        -0.99867004, -0.04681593],
       [ 0.74064575,  0.88828998,  0.34665138, ...,  0.44263362,
        -1.23654096, -0.95356649]]), sa=SampleAttributesCollection(items=[ArrayCollectable(name='chunks', doc=None, value=array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2.,
       2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
       2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
       2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
       3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
       3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 4.,
       4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4.,
       4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4.,
       4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
       2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
       2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 3., 3.,
       3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
       3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
       3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 4., 4., 4., 4., 4., 4., 4.,
       4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4.,
       4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4.,
       4., 4., 4., 4., 4.]), length=464), ArrayCollectable(name='targets', doc=None, value=array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2.,
       2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
       2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
       2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
       2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
       2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
       2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
       2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
       2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
       2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
       2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
       2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
       2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
       2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
       2., 2., 2., 2., 2.]), length=464)]), fa=FeatureAttributesCollection(items=[ArrayCollectable(name='voxel_indices', doc=None, value=array([[ 13,  60,  45],
       [ 13,  60,  46],
       [ 13,  61,  43],
       ...,
       [106,  71,  38],
       [106,  71,  39],
       [106,  71,  40]]), length=380789)]), a=DatasetAttributesCollection(items=[Collectable(name='mapper', doc=None, value=ChainMapper(nodes=[FlattenMapper(shape=(121, 145, 121), auto_train=True, space='voxel_indices'), StaticFeatureSelection(dshape=(2122945,), slicearg=array([False, False, False, ..., False, False, False])), ZScoreMapper(chunks_attr='chunks')])), Collectable(name='imgaffine', doc=None, value=array([[  -1.5,    0. ,    0. ,   90. ],
       [   0. ,    1.5,    0. , -126. ],
       [   0. ,    0. ,    1.5,  -72. ],
       [   0. ,    0. ,    0. ,    1. ]])), Collectable(name='voxel_eldim', doc=None, value=(1.5, 1.5)), Collectable(name='imghdr', doc=None, value={'session_error': array(0, dtype=int16), 'extents': array(0, dtype=int32), 'sizeof_hdr': array(348, dtype=int32), 'srow_x': array([-1.5,  0. ,  0. , 90. ], dtype=float32), 'srow_y': array([   0. ,    1.5,    0. , -126. ], dtype=float32), 'pixdim': array([-1. ,  1.5,  1.5,  1.5,  0. ,  0. ,  0. ,  0. ], dtype=float32), 'slice_start': array(0, dtype=int16), 'intent_p1': array(0., dtype=float32), 'cal_max': array(0., dtype=float32), 'xyzt_units': array(10, dtype=uint8), 'intent_p2': array(0., dtype=float32), 'intent_p3': array(0., dtype=float32), 'qoffset_x': array(90., dtype=float32), 'intent_code': array(0, dtype=int16), 'qoffset_z': array(-72., dtype=float32), 'sform_code': array(2, dtype=int16), 'cal_min': array(0., dtype=float32), 'scl_slope': 1.0, 'slice_code': array(0, dtype=uint8), 'slice_duration': array(0., dtype=float32), 'quatern_b': array(0., dtype=float32), 'hdrtype': 'Nifti1Header', 'bitpix': array(16, dtype=int16), 'descrip': array('NIFTI-1 Image', dtype='|S80'), 'glmin': array(0, dtype=int32), 'dim_info': array(0, dtype=uint8), 'glmax': array(0, dtype=int32), 'quatern_c': array(1., dtype=float32), 'data_type': array('', dtype='|S10'), 'aux_file': array('', dtype='|S24'), 'intent_name': array('', dtype='|S16'), 'vox_offset': array(0., dtype=float32), 'srow_z': array([  0. ,   0. ,   1.5, -72. ], dtype=float32), 'db_name': array('', dtype='|S18'), 'scl_inter': 0.0, 'quatern_d': array(0., dtype=float32), 'dim': array([  3, 121, 145, 121,   1,   1,   1,   1], dtype=int16), 'magic': array('n+1', dtype='|S4'), 'datatype': array(512, dtype=int16), 'regular': array('r', dtype='|S1'), 'slice_end': array(0, dtype=int16), 'qform_code': array(2, dtype=int16), 'qoffset_y': array(-126., dtype=float32), 'toffset': array(0., dtype=float32)}), Collectable(name='imgtype', doc=None, value='Nifti1Image'), Collectable(name='voxel_dim', doc=None, value=(121, 145, 121))]))>