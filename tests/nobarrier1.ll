define void @nobarrier1() {

a:
  br label %b

b:
  ret void
}