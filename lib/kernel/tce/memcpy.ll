

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define void @__pocl_tce_memcpy_p1_p2_i32(ptr addrspace(1) nocapture noundef writeonly %dst, ptr addrspace(2) nocapture noundef readonly %src, i32 noundef %bytes, i1 noundef %unused) local_unnamed_addr #0 {
entry:
  %cmp5.not = icmp eq i32 %bytes, 0
  br i1 %cmp5.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body, %entry
  %i.06 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i8, ptr addrspace(2) %src, i32 %i.06
  %0 = load i8, ptr addrspace(2) %arrayidx, align 1
  %arrayidx1 = getelementptr inbounds i8, ptr addrspace(1) %dst, i32 %i.06
  store i8 %0, ptr addrspace(1) %arrayidx1, align 1
  %inc = add nuw i32 %i.06, 1
  %exitcond.not = icmp eq i32 %inc, %bytes
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
