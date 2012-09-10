# clang.am - CLANG rules for automake projects
#
# Copyright (c) 2012 Vincent Danjean <Vincent.Danjean@ens-lyon.org>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#

# Some variables to compile with CLANG in non verbose mode
AM_V_CLANG_C = $(AM_V_CLANG_C_@AM_V@)
AM_V_CLANG_C_ = $(AM_V_CLANG_C_@AM_DEFAULT_V@)
AM_V_CLANG_C_0 = @echo "  CLANG  [C] $@";

AM_V_CLANG_CL = $(AM_V_CLANG_CL_@AM_V@)
AM_V_CLANG_CL_ = $(AM_V_CLANG_CL_@AM_DEFAULT_V@)
AM_V_CLANG_CL_0 = @echo "  CLANG  [CL] $@";

CLANG_COMPILE = $(CLANG) $(CLANGFLAGS) $(CLANG_DEFS) \
	$(CLANG_DEFAULT_INCLUDES) $(CLANG_INCLUDES) \
	$(AM_CLANG_CPPFLAGS) $(CLANG_CPPFLAGS)

CLANG_CL_COMPILE = $(CLANG_COMPILE) $(AM_CLANG_CLFLAGS) $(CLANG_CLFLAGS)
CLANG_C_COMPILE  = $(CLANG_COMPILE) $(AM_CLANG_CFLAGS)  $(CLANG_CFLAGS)

AM_V_LLAS = $(AM_V_LLAS_@AM_V@)
AM_V_LLAS_ = $(AM_V_LLAS_@AM_DEFAULT_V@)
AM_V_LLAS_0 = @echo "  LLVM   [AS] $@";

LLAS_COMPILE = $(LLVM_AS) $(LLVMFLAGS) $(LLVM_DEFS) \
	$(LLVM_DEFAULT_INCLUDES) $(LLVM_INCLUDES) \
        $(AM_LLVM_CPPFLAGS) $(LLVM_CPPFLAGS) $(AM_LLVM_ASFLAGS) $(LLVM_ASFLAGS)


# We include too much dependency files, but we cannot do better without
# proper automake support (that generate explicitly all these includes based
# on sources)
-include ./$(DEPDIR)/*.Po

.SUFFIX: .cl .ll

.cl.o:
	$(AM_V_CLANG_CL)$(CLANG_CL_COMPILE) -MT $@ -MD -MP -MF $(DEPDIR)/$*.Tpo -c -o $@ $<
	$(AM_V_at)$(am__mv) $(DEPDIR)/$*.Tpo $(DEPDIR)/$*.Po

.c.o:
	$(AM_V_CLANG_C)$(CLANG_C_COMPILE) -MT $@ -MD -MP -MF $(DEPDIR)/$*.Tpo -c -o $@ $<
	$(AM_V_at)$(am__mv) $(DEPDIR)/$*.Tpo $(DEPDIR)/$*.Po

# no support for "-MT $@ -MD -MP -MF $(DEPDIR)/$*.Tpo -c"
.ll.o:
	$(AM_V_LLAS)$(LLAS_COMPILE) -o $@ $<


