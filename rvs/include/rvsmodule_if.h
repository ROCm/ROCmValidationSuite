/********************************************************************************
 *
 * Copyright (c) 2018 ROCm Developer Tools
 *
 * MIT LICENSE:
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do
 * so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef RVS_INCLUDE_RVSMODULE_IF_H_
#define RVS_INCLUDE_RVSMODULE_IF_H_

extern "C" {
extern  int   rvs_module_init(void*);
extern  int   rvs_module_terminate(void);
extern  void* rvs_module_action_create(void);
extern  int   rvs_module_action_destroy(void*);
extern  int   rvs_module_has_interface(int);

// define function pointer types to ease late binding usage
typedef int   (*t_rvs_module_init)(void*);
typedef int   (*t_rvs_module_terminate)(void);
typedef void* (*t_rvs_module_action_create)(void);
typedef int   (*t_rvs_module_action_destroy)(void*);
typedef int   (*t_rvs_module_has_interface)(int);

}

#endif  // RVS_INCLUDE_RVSMODULE_IF_H_
