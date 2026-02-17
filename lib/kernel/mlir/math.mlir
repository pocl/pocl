
module {
  func.func private @_Z7_cl_expf(%a: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %res = math.exp %a : f32
    return %res : f32
  }
}

module {
  func.func private @_Z7_cl_logf(%a: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %res = math.log %a : f32
    return %res : f32
  }
}

module {
  func.func private @_Z9_cl_rsqrtf(%a: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %res = math.rsqrt %a : f32
    return %res : f32
  }
}

module {
  func.func private @_Z8_cl_sqrtd(%a: f64) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %res = math.sqrt %a : f64
    return %res : f64
  }
}

module {
  func.func private @_Z8_cl_sqrtf(%a: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %res = math.sqrt %a : f32
    return %res : f32
  }
}

module {
  func.func private @_Z9_cl_floorf(%a: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %res = math.floor %a : f32
    return %res : f32
  }
}

module {
  func.func private @_Z7_cl_sinf(%a: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %res = math.sin %a : f32
    return %res : f32
  }
}

module {
  func.func private @_Z7_cl_cosf(%a: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %res = math.cos %a : f32
    return %res : f32
  }
}

module {
  func.func private @_Z7_cl_sind(%a: f64) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %res = math.sin %a : f64
    return %res : f64
  }
}

module {
  func.func private @_Z8_cl_acosd(%a: f64) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %res = math.acos %a : f64
    return %res : f64
  }
}

module {
  func.func private @_Z7_cl_cosd(%a: f64) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %res = math.cos %a : f64
    return %res : f64
  }
}

module {
  func.func private @_Z7_cl_logd(%a: f64) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %res = math.log %a : f64
    return %res : f64
  }
}

module {
  func.func private @_Z7_cl_abss(%a: i16) -> i16 attributes {llvm.linkage = #llvm.linkage<external>} {
    %res = math.absi %a : i16
    return %res : i16
  }
}


module {
  func.func private @_Z11_cl_atan2piff(%y: f32, %x: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %pi = arith.constant 3.141592653589793238462 : f32
    %tmp = math.atan2 %y, %x : f32
    %res = arith.divf %tmp, %pi : f32
    return %res : f32
  }
}

module {
  func.func private @_Z9_cl_clampfff(%x: f32, %minval: f32, %maxval: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %tmp = arith.maximumf %x, %minval : f32
    %res = arith.minimumf %tmp, %maxval : f32
    return %res : f32
  }
}
