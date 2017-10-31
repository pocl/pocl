// Remove this attribute to reproduce #548, which should get fixed after
// upgrading to Clang 6.0:
__attribute__((convergent))
inline
void
auxfunc()
{
    printf("auxfunc\n");
    barrier(CLK_LOCAL_MEM_FENCE); // XXXX
}


__kernel
void
test_kernel() {
    int id = get_local_id(0);
    int localsize = get_local_size(0);
    __local int x;

    // For a workgroup size of 2, we expect CCC to be printed twice,
    // with id=0 and id=1 but the printfs have id=0 for both work-items.
    // Comment out any one or more lines marked XXXX and it works as expected.

#define TESTID1 // XXXX
#define TESTID2 // XXXX

#ifdef TESTID1
    bool test1 = (id == 0);
#else
    bool test1 = true;
#endif
#ifdef TESTID2
    bool test2 = (id == 0);
#else
    bool test2 = true;
#endif

    printf("id=%d: AAA.\n", id);
    if (test1) {
      x += 1; // XXXX
      //x = 1; // uncomment this line and it also fixes it
    }
    printf("id=%d: CCC.\n", id);
    auxfunc();
    if (test2) {
      x = 1; // XXXX
    }
    printf("id=%d: EEE.\n", id);
}
