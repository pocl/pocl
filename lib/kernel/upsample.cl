/* OpenCL built-in library: upsample()

   Copyright (c) 2011 Erik Schnetter <eschnetter@perimeterinstitute.ca>
                      Perimeter Institute for Theoretical Physics
   
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   
   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.
   
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

// We can't use templates.h because they don't allow us to create the
// convert_* function calls

#define IMPLEMENT_UPSAMPLE_LG_GUG(GTYPE, SGTYPE, UGTYPE, LGTYPE)        \
  LGTYPE _CL_OVERLOADABLE                                               \
  upsample(GTYPE a, UGTYPE b)                                           \
  {                                                                     \
    int bits = CHAR_BIT * sizeof(SGTYPE);                               \
    return (convert_##LGTYPE(a) << (LGTYPE)bits) | convert_##LGTYPE(b); \
  }

IMPLEMENT_UPSAMPLE_LG_GUG(char    , char  , uchar   , short   )
IMPLEMENT_UPSAMPLE_LG_GUG(char2   , char  , uchar2  , short2  )
IMPLEMENT_UPSAMPLE_LG_GUG(char3   , char  , uchar3  , short3  )
IMPLEMENT_UPSAMPLE_LG_GUG(char4   , char  , uchar4  , short4  )
IMPLEMENT_UPSAMPLE_LG_GUG(char8   , char  , uchar8  , short8  )
IMPLEMENT_UPSAMPLE_LG_GUG(char16  , char  , uchar16 , short16 )
IMPLEMENT_UPSAMPLE_LG_GUG(uchar   , uchar , uchar   , ushort  )
IMPLEMENT_UPSAMPLE_LG_GUG(uchar2  , uchar , uchar2  , ushort2 )
IMPLEMENT_UPSAMPLE_LG_GUG(uchar3  , uchar , uchar3  , ushort3 )
IMPLEMENT_UPSAMPLE_LG_GUG(uchar4  , uchar , uchar4  , ushort4 )
IMPLEMENT_UPSAMPLE_LG_GUG(uchar8  , uchar , uchar8  , ushort8 )
IMPLEMENT_UPSAMPLE_LG_GUG(uchar16 , uchar , uchar16 , ushort16)
IMPLEMENT_UPSAMPLE_LG_GUG(short   , short , ushort  , int     )
IMPLEMENT_UPSAMPLE_LG_GUG(short2  , short , ushort2 , int2    )
IMPLEMENT_UPSAMPLE_LG_GUG(short3  , short , ushort3 , int3    )
IMPLEMENT_UPSAMPLE_LG_GUG(short4  , short , ushort4 , int4    )
IMPLEMENT_UPSAMPLE_LG_GUG(short8  , short , ushort8 , int8    )
IMPLEMENT_UPSAMPLE_LG_GUG(short16 , short , ushort16, int16   )
IMPLEMENT_UPSAMPLE_LG_GUG(ushort  , ushort, ushort  , uint    )
IMPLEMENT_UPSAMPLE_LG_GUG(ushort2 , ushort, ushort2 , uint2   )
IMPLEMENT_UPSAMPLE_LG_GUG(ushort3 , ushort, ushort3 , uint3   )
IMPLEMENT_UPSAMPLE_LG_GUG(ushort4 , ushort, ushort4 , uint4   )
IMPLEMENT_UPSAMPLE_LG_GUG(ushort8 , ushort, ushort8 , uint8   )
IMPLEMENT_UPSAMPLE_LG_GUG(ushort16, ushort, ushort16, uint16  )
#ifdef cl_khr_int64
IMPLEMENT_UPSAMPLE_LG_GUG(int     , int   , uint    , long    )
IMPLEMENT_UPSAMPLE_LG_GUG(int2    , int   , uint2   , long2   )
IMPLEMENT_UPSAMPLE_LG_GUG(int3    , int   , uint3   , long3   )
IMPLEMENT_UPSAMPLE_LG_GUG(int4    , int   , uint4   , long4   )
IMPLEMENT_UPSAMPLE_LG_GUG(int8    , int   , uint8   , long8   )
IMPLEMENT_UPSAMPLE_LG_GUG(int16   , int   , uint16  , long16  )
IMPLEMENT_UPSAMPLE_LG_GUG(uint    , uint  , uint    , ulong   )
IMPLEMENT_UPSAMPLE_LG_GUG(uint2   , uint  , uint2   , ulong2  )
IMPLEMENT_UPSAMPLE_LG_GUG(uint3   , uint  , uint3   , ulong3  )
IMPLEMENT_UPSAMPLE_LG_GUG(uint4   , uint  , uint4   , ulong4  )
IMPLEMENT_UPSAMPLE_LG_GUG(uint8   , uint  , uint8   , ulong8  )
IMPLEMENT_UPSAMPLE_LG_GUG(uint16  , uint  , uint16  , ulong16 )
#endif
