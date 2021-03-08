// -*- C++ -*-
//===-- parallel_backend_sycl_radix_sort.h --------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_parallel_backend_sycl_radix_sort_H
#define _ONEDPL_parallel_backend_sycl_radix_sort_H

#include <CL/sycl.hpp>
#include <climits>

#include "parallel_backend_sycl_utils.h"
#include "execution_sycl_defs.h"

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{
//------------------------------------------------------------------------
// radix sort: kernel names
//------------------------------------------------------------------------

template <typename _DerivedKernelName>
class __kernel_name_base;

template <typename... _Name>
class __radix_sort_count_kernel : public __kernel_name_base<__radix_sort_count_kernel<_Name...>>
{
};

template <typename... _Name>
class __radix_sort_scan_kernel_1 : public __kernel_name_base<__radix_sort_scan_kernel_1<_Name...>>
{
};

template <typename... _Name>
class __radix_sort_scan_kernel_2 : public __kernel_name_base<__radix_sort_scan_kernel_2<_Name...>>
{
};

template <typename... _Name>
class __radix_sort_reorder_kernel : public __kernel_name_base<__radix_sort_reorder_kernel<_Name...>>
{
};

template <typename _Name>
class __odd_iteration;

//------------------------------------------------------------------------
// radix sort: ordered traits for a given size and integral/float flag
//------------------------------------------------------------------------

template <::std::size_t __type_size, bool __is_integral_type>
struct __get_ordered
{
};

template <>
struct __get_ordered<1, true>
{
    using _type = uint8_t;
    constexpr static ::std::int8_t __mask = 0x80;
};

template <>
struct __get_ordered<2, true>
{
    using _type = uint16_t;
    constexpr static ::std::int16_t __mask = 0x8000;
};

template <>
struct __get_ordered<4, true>
{
    using _type = uint32_t;
    constexpr static ::std::int32_t __mask = 0x80000000;
};

template <>
struct __get_ordered<8, true>
{
    using _type = uint64_t;
    constexpr static ::std::int64_t __mask = 0x8000000000000000;
};

template <>
struct __get_ordered<4, false>
{
    using _type = uint32_t;
    constexpr static ::std::uint32_t __nmask = 0xFFFFFFFF; // for negative numbers
    constexpr static ::std::uint32_t __pmask = 0x80000000; // for positive numbers
};

template <>
struct __get_ordered<8, false>
{
    using _type = uint64_t;
    constexpr static ::std::uint64_t __nmask = 0xFFFFFFFFFFFFFFFF; // for negative numbers
    constexpr static ::std::uint64_t __pmask = 0x8000000000000000; // for positive numbers
};

//------------------------------------------------------------------------
// radix sort: ordered type for a given type
//------------------------------------------------------------------------

// for unknown/unsupported type we do not have any trait
template <typename _T, typename _Dummy = void>
struct __ordered
{
};

// for unsigned integrals we use the same type
template <typename _T>
struct __ordered<_T, __enable_if_t<::std::is_integral<_T>::value&& ::std::is_unsigned<_T>::value>>
{
    using _type = _T;
};

// for signed integrals or floatings we map: size -> corresponding unsigned integral
template <typename _T>
struct __ordered<_T, __enable_if_t<(::std::is_integral<_T>::value && ::std::is_signed<_T>::value) ||
                                   ::std::is_floating_point<_T>::value>>
{
    using _type = typename __get_ordered<sizeof(_T), ::std::is_integral<_T>::value>::_type;
};

// shorthands
template <typename _T>
using __ordered_t = typename __ordered<_T>::_type;

//------------------------------------------------------------------------
// radix sort: functions for conversion to ordered type
//------------------------------------------------------------------------

// for already ordered types (any uints) we use the same type
template <typename _T>
inline __enable_if_t<::std::is_same<_T, __ordered_t<_T>>::value, __ordered_t<_T>>
__convert_to_ordered(_T __value)
{
    return __value;
}

// converts integral type to ordered (in terms of bitness) type
template <typename _T>
inline __enable_if_t<!::std::is_same<_T, __ordered_t<_T>>::value && !::std::is_floating_point<_T>::value,
                     __ordered_t<_T>>
__convert_to_ordered(_T __value)
{
    _T __result = __value ^ __get_ordered<sizeof(_T), true>::__mask;
    return *reinterpret_cast<__ordered_t<_T>*>(&__result);
}

// converts floating type to ordered (in terms of bitness) type
template <typename _T>
inline __enable_if_t<!::std::is_same<_T, __ordered_t<_T>>::value && ::std::is_floating_point<_T>::value,
                     __ordered_t<_T>>
__convert_to_ordered(_T __value)
{
    __ordered_t<_T> __uvalue = *reinterpret_cast<__ordered_t<_T>*>(&__value);
    // check if value negative
    __ordered_t<_T> __is_negative = __uvalue >> (sizeof(_T) * CHAR_BIT - 1);
    // for positive: 00..00 -> 00..00 -> 10..00
    // for negative: 00..01 -> 11..11 -> 11..11
    __ordered_t<_T> __ordered_mask =
        (__is_negative * __get_ordered<sizeof(_T), false>::__nmask) | __get_ordered<sizeof(_T), false>::__pmask;
    return __uvalue ^ __ordered_mask;
}

//------------------------------------------------------------------------
// radix sort: run-time device info functions
//------------------------------------------------------------------------

// get rounded up result of (__number / __divisor)
template <typename _T1, typename _T2>
inline auto
__get_roundedup_div(_T1 __number, _T2 __divisor) -> decltype((__number - 1) / __divisor + 1)
{
    return (__number - 1) / __divisor + 1;
}

//------------------------------------------------------------------------
// radix sort: bit pattern functions
//------------------------------------------------------------------------

// get number of states radix bits can represent
constexpr ::std::uint32_t
__get_states_in_bits(::std::uint32_t __radix_bits)
{
    return (1 << __radix_bits);
}

// get number of buckets (size of radix bits) in T
template <typename _T>
constexpr ::std::uint32_t
__get_buckets_in_type(::std::uint32_t __radix_bits)
{
    return (sizeof(_T) * CHAR_BIT) / __radix_bits;
}

// required for descending comparator support
template <bool __flag>
struct __invert_if
{
    template <typename _T>
    _T
    operator()(_T __value)
    {
        return __value;
    }
};

// invert value if descending comparator is passed
template <>
struct __invert_if<true>
{
    template <typename _T>
    _T
    operator()(_T __value)
    {
        return ~__value;
    }

    // invertation for bool type have to be logical, rather than bit
    bool
    operator()(bool __value)
    {
        return !__value;
    }
};

// get bit values in a certain bucket of a value
template <::std::uint32_t __radix_bits, bool __is_comp_asc, typename _T>
::std::uint32_t
__get_bucket_value(_T __value, ::std::uint32_t __radix_iter)
{
    // invert value if we need to sort in descending order
    __value = __invert_if<!__is_comp_asc>{}(__value);

    // get bucket offset idx from the end of bit type (least significant bits)
    ::std::uint32_t __bucket_offset = __radix_iter * __radix_bits;

    // get offset mask for one bucket, e.g.
    // radix_bits=2: 0000 0001 -> 0000 0100 -> 0000 0011
    __ordered_t<_T> __bucket_mask = (1u << __radix_bits) - 1u;

    // get bits under bucket mask
    return (__value >> __bucket_offset) & __bucket_mask;
}

template <typename _T, bool __is_comp_asc>
inline __enable_if_t<__is_comp_asc, _T>
__get_last_value()
{
    return ::std::numeric_limits<_T>::max();
};

template <typename _T, bool __is_comp_asc>
inline __enable_if_t<!__is_comp_asc, _T>
__get_last_value()
{
    return ::std::numeric_limits<_T>::min();
};

//-----------------------------------------------------------------------
// radix sort: count kernel (per iteration)
//-----------------------------------------------------------------------

template <typename _KernelName, ::std::uint32_t __radix_bits, ::std::uint32_t __items_per_work_item, bool __is_comp_asc,
          typename _ExecutionPolicy, typename _InRange, typename _OffsetBuf
#if _ONEDPL_COMPILE_KERNEL
          ,
          typename _Kernel
#endif
          >
sycl::event
__radix_sort_count_submit(_ExecutionPolicy&& __exec, ::std::size_t __count_work_group_size, ::std::size_t __segments,
                          ::std::uint32_t __radix_iter, _InRange&& __in_rng, _OffsetBuf& __offset_buf,
                          sycl::event __dependency_event
#if _ONEDPL_COMPILE_KERNEL
                          ,
                          _Kernel& __kernel
#endif
)
{
    // typedefs
    using _InputT = oneapi::dpl::__internal::__value_t<_InRange>;
    using _OffsetT = typename _OffsetBuf::value_type;

    // radix states used for an array storing bucket state counters
    const ::std::uint32_t __radix_states = __get_states_in_bits(__radix_bits);
    const ::std::size_t __n = __in_rng.size();

    sycl::event __count_event = __exec.queue().submit([&](sycl::handler& __hdl) {
        __hdl.depends_on(__dependency_event);

        oneapi::dpl::__ranges::__require_access(__hdl, __in_rng); //get an access to data under SYCL buffer
        auto __offset_acc = __offset_buf.template get_access<sycl::access::mode::write>(__hdl);
        sycl::accessor<_OffsetT, 1, sycl::access::mode::read_write, sycl::access::target::local> __offset_local_acc(__radix_states, __hdl);
        __hdl.parallel_for<_KernelName>(
#if _ONEDPL_COMPILE_KERNEL
            __kernel,
#endif
            sycl::nd_range<1>(__segments * __count_work_group_size, __count_work_group_size), [=](sycl::nd_item<1> __item) {
                ::std::size_t __group_id = __item.get_group(0);
                ::std::size_t __local_id = __item.get_local_id(0);
                ::std::size_t __global_id = __item.get_global_id(0);

                if(__count_work_group_size < __radix_states)
                {
                    for(::std::uint32_t __radix_state = 0; __radix_state < __radix_states; ++__radix_state)
                        __offset_local_acc[__radix_state] = 0;
                }
                else
                {
                    if (__local_id < __radix_states)
                        __offset_local_acc[__local_id] = 0;
                }

                for(::std::uint32_t __iter = 0; __iter < __items_per_work_item; ++__iter)
                {
                    ::std::size_t __group_start_id = __group_id * __items_per_work_item * __count_work_group_size;
                    ::std::size_t __iter_start_id = __iter * __count_work_group_size;
                    ::std::size_t __adjusted_id = __group_start_id + __iter_start_id + __local_id;

                    bool __is_valid = __adjusted_id < __n;
                    __ordered_t<_InputT> __val = __is_valid ? __convert_to_ordered(__in_rng[__adjusted_id]) : __ordered_t<_InputT>{};
                    ::std::uint32_t __bucket = __get_bucket_value<__radix_bits, __is_comp_asc>(__val, __radix_iter);

                    for(::std::uint32_t __radix_state = 0; __radix_state < __radix_states; ++__radix_state)
                    {
                        _OffsetT __is_current_bucket = __is_valid ? (__bucket == __radix_state) : false;
                        _OffsetT __items_in_bucket =
                            sycl::ONEAPI::reduce(__item.get_group(), __is_current_bucket, sycl::ONEAPI::plus<_OffsetT>());
                        if (__local_id == 0)
                            __offset_local_acc[__radix_state] += __items_in_bucket;
                    }
                }

                __item.barrier(sycl::access::fence_space::local_space);
                if(__count_work_group_size < __radix_states)
                {
                    for(::std::uint32_t __radix_state = 0; __radix_state < __radix_states; ++__radix_state)
                        __offset_acc[(__segments + 1) * __radix_state + __group_id] = __offset_local_acc[__radix_state];
                }
                else
                {
                    if (__local_id < __radix_states)
                        __offset_acc[(__segments + 1) * __local_id + __group_id] = __offset_local_acc[__local_id];
                }
        });
    });
    return __count_event;
}

//-----------------------------------------------------------------------
// radix sort: scan kernel (per iteration)
//-----------------------------------------------------------------------

template <typename _KernelName1, typename _KernelName2, ::std::uint32_t __radix_bits, ::std::uint32_t __items_per_work_item, typename _ExecutionPolicy,
          typename _OffsetBuf>
sycl::event
__radix_sort_scan_submit(_ExecutionPolicy&& __exec, ::std::size_t __scan_work_group_size, ::std::size_t __segments,
                         _OffsetBuf& __offset_buf, sycl::event __dependency_event)
{
    using _Offset_T = typename _OffsetBuf::value_type;

    // there are no local offsets for the first segment, but the rest segments shoud be scanned
    // with respect to the count value in the first segment what requires n + 1 positions
    const ::std::size_t __scan_size = __segments + 1;

    __scan_work_group_size = ::std::min(__scan_size, __scan_work_group_size);

    const ::std::uint32_t __radix_states = __get_states_in_bits(__radix_bits);
    const ::std::size_t __global_scan_begin = __offset_buf.get_count() - __radix_states;

    // 1. Local scan: produces local offsets using count values
    sycl::event __scan_event = __exec.queue().submit([&](sycl::handler& __hdl) {
        __hdl.depends_on(__dependency_event);
        auto __offset_acc = __offset_buf.template get_access<sycl::access::mode::read_write>(__hdl);
        __hdl.parallel_for<_KernelName1>(
            sycl::nd_range<1>(__radix_states * __scan_work_group_size, __scan_work_group_size), [=](sycl::nd_item<1> __self_item) {
                // find borders of a region with a specific bucket id
                sycl::global_ptr<_Offset_T> __begin = __offset_acc.get_pointer() + __scan_size * __self_item.get_group(0);
                // TODO: consider another approach with use of local memory
                sycl::ONEAPI::exclusive_scan(__self_item.get_group(), __begin, __begin + __scan_size, __begin,
                                             _Offset_T(0), sycl::ONEAPI::plus<_Offset_T>{});
            });
    });

    // 2. Global scan: produces global offsets using local offsets
    __scan_event = __exec.queue().submit([&](sycl::handler& __hdl) {
        __hdl.depends_on(__scan_event);
        auto __offset_acc = __offset_buf.template get_access<sycl::access::mode::read_write>(__hdl);
        __hdl.parallel_for<_KernelName2>(
            sycl::nd_range<1>(__radix_states, __radix_states), [=](sycl::nd_item<1> __self_item) {
                ::std::size_t __self_lidx = __self_item.get_local_id(0);

                ::std::size_t __global_offset_idx = __global_scan_begin + __self_lidx;
                ::std::size_t __last_segment_bucket_idx = (__self_lidx + 1) * __scan_size - 1;

                // copy buckets from the last segment, scan them to get global offsets
                __offset_acc[__global_offset_idx] = sycl::ONEAPI::exclusive_scan(
                    __self_item.get_group(), __offset_acc[__last_segment_bucket_idx], sycl::ONEAPI::plus<_Offset_T>{});
            });
    });

    return __scan_event;
}

//-----------------------------------------------------------------------
// radix sort: a function for reorder phase of one iteration
//-----------------------------------------------------------------------

template <typename _KernelName, ::std::uint32_t __radix_bits, ::std::uint32_t __items_per_work_item, bool __is_comp_asc, typename _ExecutionPolicy,
          typename _InRange, typename _OutRange, typename _OffsetBuf
#if _ONEDPL_COMPILE_KERNEL
          ,
          typename _Kernel
#endif
          >
sycl::event
__radix_sort_reorder_submit(_ExecutionPolicy&& __exec, ::std::size_t __reorder_work_group_size, ::std::size_t __segments,
                            ::std::uint32_t __radix_iter, _InRange&& __in_rng,
                            _OutRange&& __out_rng, _OffsetBuf& __offset_buf, sycl::event __dependency_event
#if _ONEDPL_COMPILE_KERNEL
                            ,
                            _Kernel& __kernel
#endif
)
{
    // typedefs
    using _InputT = oneapi::dpl::__internal::__value_t<_InRange>;
    using _OffsetT = typename _OffsetBuf::value_type;

    const ::std::size_t __n = __out_rng.size();

    // submit to reorder values
    sycl::event __reorder_event = __exec.queue().submit([&](sycl::handler& __hdl) {
        __hdl.depends_on(__dependency_event);
        oneapi::dpl::__ranges::__require_access(__hdl, __in_rng, __out_rng);
        auto __offset_acc = __offset_buf.template get_access<sycl::access::mode::read>(__hdl);
        __hdl.parallel_for<_KernelName>(
#if _ONEDPL_COMPILE_KERNEL
            __kernel,
#endif
            sycl::nd_range<1>(__segments * __reorder_work_group_size, __reorder_work_group_size), [=](sycl::nd_item<1> __item) {
                ::std::size_t __global_id = __item.get_global_id(0);
            });
    });

    return __reorder_event;
}

//-----------------------------------------------------------------------
// radix sort: a function for one iteration
//-----------------------------------------------------------------------

template <::std::uint32_t __radix_bits, ::std::uint32_t __items_per_work_item, bool __is_comp_asc, typename _ExecutionPolicy, typename _InRange,
          typename _OutRange, typename _TmpBuf>
sycl::event
__parallel_radix_sort_iteration(_ExecutionPolicy&& __exec, ::std::size_t __segments, ::std::uint32_t __radix_iter,
                                _InRange&& __in_rng, _OutRange&& __out_rng, _TmpBuf&& __offset_buf,
                                sycl::event __dependency_event)
{
    using _CustomName = typename __decay_t<_ExecutionPolicy>::kernel_name;
    using _RadixCountKernel =
        oneapi::dpl::__par_backend_hetero::__internal::_KernelName_t<__radix_sort_count_kernel, _CustomName,
                                                                     __decay_t<_InRange>, __decay_t<_TmpBuf>>;
    using _RadixLocalScanKernel =
        oneapi::dpl::__par_backend_hetero::__internal::_KernelName_t<__radix_sort_scan_kernel_1, _CustomName,
                                                                     __decay_t<_TmpBuf>>;
    using _RadixGlobalScanKernel =
        oneapi::dpl::__par_backend_hetero::__internal::_KernelName_t<__radix_sort_scan_kernel_2, _CustomName,
                                                                     __decay_t<_TmpBuf>>;
    using _RadixReorderKernel =
        oneapi::dpl::__par_backend_hetero::__internal::_KernelName_t<__radix_sort_reorder_kernel, _CustomName,
                                                                     __decay_t<_InRange>, __decay_t<_OutRange>>;

    ::std::size_t __max_work_group_size = 32; // oneapi::dpl::__internal::__max_work_group_size(__exec);
    ::std::size_t __count_work_group_size = __max_work_group_size;
    ::std::size_t __scan_work_group_size = __max_work_group_size;
    ::std::size_t __reorder_work_group_size = __max_work_group_size;

#if _ONEDPL_COMPILE_KERNEL
    auto __count_kernel = _RadixCountKernel::__compile_kernel(::std::forward<_ExecutionPolicy>(__exec));
    auto __reorder_kernel = _RadixReorderKernel::__compile_kernel(::std::forward<_ExecutionPolicy>(__exec));
#endif

    // 1. Count Phase
    sycl::event __count_event = __radix_sort_count_submit<_RadixCountKernel, __radix_bits, __items_per_work_item, __is_comp_asc>(
        ::std::forward<_ExecutionPolicy>(__exec), __count_work_group_size, __segments, __radix_iter, __in_rng, __offset_buf, __dependency_event
#if _ONEDPL_COMPILE_KERNEL
        ,
        __count_kernel
#endif
    );

    // 2. Scan Phase
    sycl::event __scan_event = __radix_sort_scan_submit<_RadixLocalScanKernel, _RadixGlobalScanKernel, __radix_bits, __items_per_work_item>(
        ::std::forward<_ExecutionPolicy>(__exec), __scan_work_group_size, __segments, __offset_buf, __count_event);

//     // 3. Reorder Phase
//     sycl::event __reorder_event = __radix_sort_reorder_submit<_RadixReorderKernel, __radix_bits, __items_per_work_item, __is_comp_asc>(
//         ::std::forward<_ExecutionPolicy>(__exec), __reorder_work_group_size, __segments, __radix_iter, __in_rng, __out_rng, __offset_buf, __scan_event
// #if _ONEDPL_COMPILE_KERNEL
//         ,
//         __reorder_kernel
// #endif
//     );

    return __reorder_event;
}

//-----------------------------------------------------------------------
// radix sort: main function
//-----------------------------------------------------------------------

template <bool __is_comp_asc, ::std::uint32_t __radix_bits = 4, ::std::uint32_t __items_per_work_item = 1,
          typename _ExecutionPolicy, typename _Range>
__future<void>
__parallel_radix_sort(_ExecutionPolicy&& __exec, _Range&& __in_rng)
{
    using _DecExecutionPolicy = __decay_t<_ExecutionPolicy>;
    using _T = oneapi::dpl::__internal::__value_t<_Range>;

    const ::std::size_t __n = __in_rng.size();
    assert(__n > 1);

    const ::std::size_t __work_group_size = 32; // oneapi::dpl::__internal::__max_work_group_size(__exec);
    const ::std::size_t __segments = __get_roundedup_div(__n, __work_group_size * __items_per_work_item);

    const ::std::uint32_t __radix_iters = __get_buckets_in_type<_T>(__radix_bits);
    const ::std::uint32_t __radix_states = __get_states_in_bits(__radix_bits);

    // memory for storing count and offset values
    // additional 2 * __radix_states elements are used for getting local and global offsets from count values
    sycl::buffer<::std::uint32_t, 1> __offset_buf(sycl::range<1>(__segments * __radix_states + 2 * __radix_states));

    // memory for storing values sorted for an iteration
    __internal::__buffer<_DecExecutionPolicy, _T> __out_buffer_holder{__exec, __n};
    oneapi::dpl::__ranges::all_view<_T, sycl::access::mode::read_write> __out_rng(__out_buffer_holder.get_buffer());

    // iterations per each bucket
    assert("Number of iterations must be even" && __radix_iters % 2 == 0);
    // TODO: radix for bool can be made using 1 iteration (x2 speedup against current implementation)
    sycl::event __iteration_event{};
    for (::std::uint32_t __radix_iter = 0; __radix_iter < __radix_iters; ++__radix_iter)
    {
        // TODO: convert to ordered type once at the first iteration and convert back at the last one
        if (__radix_iter % 2 == 0)
            __iteration_event = __parallel_radix_sort_iteration<__radix_bits, __items_per_work_item, __is_comp_asc>(
                ::std::forward<_ExecutionPolicy>(__exec), __segments, __radix_iter, __in_rng, __out_rng, __offset_buf,
                __iteration_event);
        else //swap __in_rng and __out_rng
            __iteration_event = __parallel_radix_sort_iteration<__radix_bits, __items_per_work_item, __is_comp_asc>(
                make_wrapped_policy<__odd_iteration>(::std::forward<_ExecutionPolicy>(__exec)), __segments,
                __radix_iter, __out_rng, __in_rng, __offset_buf, __iteration_event);

        // TODO: since reassign to __iteration_event does not work, we have to make explicit wait on the event
        explicit_wait_if<::std::is_pointer<decltype(__in_rng.begin())>::value>{}(__iteration_event);
    }

    return __future<void>(__iteration_event, __out_buffer_holder.get_buffer());
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif /* _ONEDPL_parallel_backend_sycl_radix_sort_H */
