## HIP backend build

1. export the LD_LIBRARY_PATH for rocblas.<br/>
    `export LD_LIBRARY_PATH=/opt/rocm-5.3.0/lib/rocblas/library:$LD_LIBRARY_PATH`
2. export the LIBRARY_PATH.<br/>
    `export LIBRARY_PATH=/opt/rocm-5.3.0/lib:$LIBRARY_PATH`
3. build lc0. <br/>
    `CC=hipcc CXX=hipcc ./build.sh -Dhip=true`

