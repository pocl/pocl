; ModuleID = 'subgroup.bc'
source_filename = "subgroup.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: convergent mustprogress nounwind
define void @_Z17sub_group_barrieri(i32 noundef %0) local_unnamed_addr #0 {
  tail call void @llvm.nvvm.bar.warp.sync(i32 -1)
  %2 = and i32 %0, 1
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %5, label %4

4:                                                ; preds = %1
  tail call void @llvm.nvvm.membar.cta()
  br label %5

5:                                                ; preds = %4, %1
  %6 = and i32 %0, 2
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %9, label %8

8:                                                ; preds = %5
  tail call void @llvm.nvvm.membar.gl()
  br label %9

9:                                                ; preds = %8, %5
  ret void
}

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.bar.warp.sync(i32) #1

; Function Attrs: nocallback nounwind
declare void @llvm.nvvm.membar.cta() #2

; Function Attrs: nocallback nounwind
declare void @llvm.nvvm.membar.gl() #2

; Function Attrs: convergent mustprogress nounwind
define noundef <4 x i32> @_Z16sub_group_balloti(i32 noundef %0) local_unnamed_addr #3 {
  %2 = icmp ne i32 %0, 0
  %3 = tail call i32 @llvm.nvvm.vote.ballot(i1 %2)
  %4 = insertelement <4 x i32> <i32 poison, i32 0, i32 0, i32 0>, i32 %3, i64 0
  ret <4 x i32> %4
}

; Function Attrs: convergent inaccessiblememonly nocallback nounwind
declare i32 @llvm.nvvm.vote.ballot(i1) #4

; Function Attrs: convergent mustprogress nounwind
define noundef i32 @_Z22get_sub_group_local_idv() local_unnamed_addr #0 {
  %1 = tail call noundef i64 @_Z19get_local_linear_idv() #6
  %2 = trunc i64 %1 to i32
  %3 = and i32 %2, 31
  ret i32 %3
}

; Function Attrs: convergent nounwind
declare noundef i64 @_Z19get_local_linear_idv() local_unnamed_addr #5

; Function Attrs: convergent mustprogress nounwind
define noundef i32 @_Z23intel_sub_group_shufflejj(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = tail call i32 @llvm.nvvm.shfl.idx.i32(i32 %0, i32 %1, i32 31)
  ret i32 %3
}

; Function Attrs: convergent inaccessiblememonly nocallback nounwind
declare i32 @llvm.nvvm.shfl.idx.i32(i32, i32, i32) #4

; Function Attrs: convergent mustprogress nounwind
define noundef i32 @_Z23intel_sub_group_shuffleij(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = tail call i32 @llvm.nvvm.shfl.idx.i32(i32 %0, i32 %1, i32 31)
  ret i32 %3
}

; Function Attrs: convergent mustprogress nounwind
define noundef float @_Z23intel_sub_group_shufflefj(float noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = tail call contract float @llvm.nvvm.shfl.idx.f32(float %0, i32 %1, i32 31)
  ret float %3
}

; Function Attrs: convergent inaccessiblememonly nocallback nounwind
declare float @llvm.nvvm.shfl.idx.f32(float, i32, i32) #4

; Function Attrs: convergent mustprogress nounwind
define noundef i32 @_Z27intel_sub_group_shuffle_xorij(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = tail call i32 @llvm.nvvm.shfl.bfly.i32(i32 %0, i32 %1, i32 31)
  ret i32 %3
}

; Function Attrs: convergent inaccessiblememonly nocallback nounwind
declare i32 @llvm.nvvm.shfl.bfly.i32(i32, i32, i32) #4

; Function Attrs: convergent mustprogress nounwind
define noundef i32 @_Z27intel_sub_group_shuffle_xorjj(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = tail call i32 @llvm.nvvm.shfl.bfly.i32(i32 %0, i32 %1, i32 31)
  ret i32 %3
}

; Function Attrs: convergent mustprogress nounwind
define noundef float @_Z27intel_sub_group_shuffle_xorfj(float noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = tail call contract float @llvm.nvvm.shfl.bfly.f32(float %0, i32 %1, i32 31)
  ret float %3
}

; Function Attrs: convergent inaccessiblememonly nocallback nounwind
declare float @llvm.nvvm.shfl.bfly.f32(float, i32, i32) #4

attributes #0 = { convergent mustprogress nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_70" "target-features"="+ptx75,+sm_70" }
attributes #1 = { convergent nocallback nounwind }
attributes #2 = { nocallback nounwind }
attributes #3 = { convergent mustprogress nounwind "frame-pointer"="all" "min-legal-vector-width"="128" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_70" "target-features"="+ptx75,+sm_70" }
attributes #4 = { convergent inaccessiblememonly nocallback nounwind }
attributes #5 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_70" "target-features"="+ptx75,+sm_70" }
attributes #6 = { convergent nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4, !5}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 11, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!3 = !{i32 7, !"frame-pointer", i32 2}
!4 = !{!"Ubuntu clang version 15.0.7"}
!5 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
