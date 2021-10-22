//==- partition_array.hpp --- SYCL Xilinx array partition extension  -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains a class expressing arrays that can be partitioned.
///
/// \todo While Xilinx xocc supports multidimensional C arrays, the
/// current implementation only support 1-dim array partition.
///    .
/// \todo Extend this with multidimensional C++ arrays, such as with future
/// mdspan C++20 syntax.
///
//===----------------------------------------------------------------------===//

#ifndef SYCL_XILINX_FPGA_PARTITION_ARRAY_HPP
#define SYCL_XILINX_FPGA_PARTITION_ARRAY_HPP

#include "CL/sycl/detail/defines.hpp"
//#include "llvm/CodeGen/ISDOpcodes.h"

#include <array>
#include <cstddef>
#include <initializer_list>
#include <tuple>
#include <type_traits>
#include <utility>

namespace sycl::ext::xilinx {

namespace Repartition {
struct NoScatter {
private:
  template <size_t DimIdx, size_t ProdUntilNow, size_t totalDim, size_t HeadDim,
            size_t... DimSize>
  static constexpr size_t getOffset(std::array<size_t, totalDim> &index) {
    constexpr size_t CurProd = ProdUntilNow / HeadDim;
    size_t CurDimOffset = CurProd * index[DimIdx];
    if constexpr (sizeof...(DimSize) == 0) {
      return CurDimOffset;
    } else {
      return CurDimOffset +
             getOffset<DimIdx + 1, CurProd, totalDim, DimSize...>(index);
    }
  }

public:
  template <size_t... DimSize>
  static constexpr size_t get_nb_banks(std::index_sequence<DimSize...>) {
    return 1;
  }

  template <size_t... DimSize>
  static constexpr size_t get_bank_size(std::index_sequence<DimSize...>) {
    return (DimSize * ...);
  }

  template <size_t... DimSize>
  static constexpr std::tuple<size_t, size_t>
  get_map_for(std::index_sequence<DimSize...> dim,
              std::array<size_t, sizeof...(DimSize)> index) {
    constexpr size_t bank_idx = 0;
    size_t offset =
        getOffset<0, get_bank_size(dim), sizeof...(DimSize), DimSize...>(index);
    return std::make_tuple(bank_idx, offset);
  }

  template <size_t... DimSize>
  static constexpr bool is_valid(std::index_sequence<DimSize...>) {
    return (DimSize * ...) != 0;
  }
};

struct Complete {
private:
  template <size_t DimIdx, size_t ProdUntilNow, size_t totalDim, size_t HeadDim,
            size_t... DimSize>
  static constexpr size_t getOffset(std::array<size_t, totalDim> &index) {
    constexpr size_t CurProd = ProdUntilNow / HeadDim;
    size_t CurDimOffset = CurProd * index[DimIdx];
    if constexpr (sizeof...(DimSize) == 0) {
      return CurDimOffset;
    } else {
      return CurDimOffset +
             getOffset<DimIdx + 1, CurProd, totalDim, DimSize...>(index);
    }
  }

public:
  template <size_t... DimSize>
  static constexpr size_t get_nb_banks(std::index_sequence<DimSize...>) {
    return (DimSize * ...);
  }

  template <size_t... DimSize>
  static constexpr size_t get_bank_size(std::index_sequence<DimSize...>) {
    return 1;
  }

  template <size_t... DimSize>
  static constexpr std::tuple<size_t, size_t>
  get_map_for(std::index_sequence<DimSize...> dim,
              std::array<size_t, sizeof...(DimSize)> index) {
    size_t bank_idx =
        getOffset<0, get_bank_size(dim), sizeof...(DimSize), DimSize...>(index);
    ;
    constexpr size_t offset = 0;

    return std::make_tuple(bank_idx, offset);
  }

  template <size_t... DimSize>
  static constexpr bool is_valid(std::index_sequence<DimSize...>) {
    return (DimSize * ...) != 0;
  }
};
} // namespace Repartition

template <typename ElementType, typename RepartitionPolicy, size_t... Dims>
class PartitionnedArray {
public:
  using dim_t = std::index_sequence<Dims...>;
  static constexpr size_t NbDims = sizeof...(Dims);
  static_assert(NbDims > 0, "PartitionnedArray needs at least one dimension");
  static constexpr size_t NBBanks = RepartitionPolicy::get_nb_banks(dim_t{});
  static constexpr size_t BankSize = RepartitionPolicy::get_bank_size(dim_t{});

private:
  using idx_t = std::array<size_t, NbDims>;

  template <typename T, typename Seq> struct expander;

  template <typename T, std::size_t... Is>
  struct expander<T, std::index_sequence<Is...>> {
    template <typename E, std::size_t> using elem = E;

    using type = std::tuple<elem<T, Is>...>;
  };
  template <size_t CurDim, bool is_const> class AccessProxy {
  private:
    using RefHolderType =
        typename expander<size_t, std::make_index_sequence<CurDim>>::type;
    using PartitionnedArrayRef = std::conditional_t<is_const, PartitionnedArray const &,PartitionnedArray &>;
    PartitionnedArrayRef Accessed;
    RefHolderType DimIDX;
    static constexpr bool IsLast = CurDim == NbDims - 1;

    using SuccessorType = std::conditional_t<
        IsLast, 
        std::conditional_t<
            is_const, 
                ElementType, 
                ElementType &>,
        AccessProxy<CurDim + 1, is_const>>;

  public:
    AccessProxy(PartitionnedArrayRef _Accessed, RefHolderType _DimIDX)
        : Accessed{_Accessed}, DimIDX{_DimIDX} {}

    constexpr SuccessorType operator[](size_t Idx) {
      auto IdxSeq = std::tuple_cat(DimIDX, std::make_tuple(Idx));
      if constexpr (IsLast) {
        constexpr auto GetArray = [](auto&& ... x){ return std::array{std::forward<decltype(x)>(x) ... }; };
        return Accessed.get(std::apply(GetArray, IdxSeq));
      } else {
        return SuccessorType{Accessed, IdxSeq};
      }
    }
  };

  template <size_t DimIdx, size_t ProdUntilNow, size_t HeadDim,
            size_t... DimSize>
  static constexpr size_t getOffset(idx_t index) {
    constexpr size_t CurProd = ProdUntilNow / HeadDim;
    size_t CurDimOffset = CurProd * index[DimIdx];
    if constexpr (sizeof...(DimSize) == 0) {
      return CurDimOffset;
    } else {
      return CurDimOffset + getOffset<DimIdx + 1, CurProd, DimSize...>(index);
    }
  }

  /// @brief Construct parallel_for equivalent loop recursively
  ///
  /// @tparam CurDimIdx Which dimension of the iteration space (starting from 0)
  ///
  /// @param ElementWiseOp The kernel that should be applied to each element
  /// @param OuterLoopIterIdx array storing loop idx
  ///
  /// Build one loop of depth CurDimIDX for the loop nest corresponding to
  /// IterDomain
  template <int CurDimIdx>
  inline void build_loop_nest_rec(auto ElementWiseOp, idx_t &OuterLoopIterIdx) {
    size_t &CurDimIter = OuterLoopIterIdx[CurDimIdx];
    constexpr std::array<size_t, sizeof...(Dims)> IterDomain = {Dims...};
    for (CurDimIter = 0; CurDimIter < IterDomain[CurDimIdx]; CurDimIter++) {
      if constexpr (CurDimIdx + 1 >= NbDims) {
        static_assert(CurDimIdx == NbDims - 1);
        // innermost loop : perform the call
        ElementWiseOp(OuterLoopIterIdx);
      } else {
        // Extend the iteration index tuple and generate the next loop level
        build_loop_nest_rec<CurDimIdx + 1>(ElementWiseOp, OuterLoopIterIdx);
      }
    }
  }

  inline void apply_on_all_elem(auto ElementWiseOp) {
    idx_t idx;
    build_loop_nest_rec<0>(ElementWiseOp, idx);
  }
  using bank = std::array<ElementType, BankSize>;
  using bank_collection =
      typename expander<bank, std::make_index_sequence<NBBanks>>::type;

  bank_collection banks;

  template <size_t stage>
  constexpr ElementType const &get_elem_req(size_t bank_id, size_t offset) const {
    if constexpr (stage == NBBanks - 1) {
      return std::get<stage>(banks)[offset];
    } else {
      return (bank_id == stage) ? std::get<stage>(banks)[offset]
                                : get_elem_req<stage + 1>(bank_id, offset);
    }
  }

  constexpr ElementType const &get_elem(size_t bank_id, size_t offset) const {
    return get_elem_req<0>(bank_id, offset);
  }

  constexpr ElementType &get_elem(size_t bank_id, size_t offset) {
    return const_cast<ElementType &>(
        static_cast<PartitionnedArray const &>(*this).get_elem_req<0>(bank_id,
                                                                      offset));
  }

public:
  constexpr ElementType const & get(idx_t index) const {
    auto [bank_id, offset] = RepartitionPolicy::get_map_for(dim_t{}, index);
    return get_elem(bank_id, offset);
  }

  constexpr ElementType &get(idx_t index) {
    return const_cast<ElementType &>(
        static_cast<PartitionnedArray const &>(*this).get(index));
  }

  constexpr PartitionnedArray(std::initializer_list<ElementType> initList) {
    static_assert(RepartitionPolicy::is_valid(dim_t{}));
    auto iter = initList.begin();
    apply_on_all_elem([&](idx_t idx) {
      auto [bank_id, offset] = RepartitionPolicy::get_map_for(dim_t{}, idx);
      auto in_offset = getOffset<0, (Dims * ...), Dims...>(idx);
      get_elem(bank_id, offset) = *iter++;
    });
  }

  constexpr auto operator[](size_t value) {
      return AccessProxy<0, false>(*this, {})[value];
  }

  constexpr auto operator[](size_t value) const {
      return AccessProxy<0, true>(*this, {})[value];
  }
};
} // namespace sycl::ext::xilinx
#if 0
__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl::ext::xilinx {

/** Kind of array partition

    To be used when defining or declaring a vendor-supported partition
    array in kernel.
*/
namespace partition {
  /** Three real partition types: cyclic, block, complete.

      none represents non partitioned standard array.
  */
  namespace type {
  enum type : int {
    cyclic,
    block,
    complete,
    none
  };
  }

  /// This fuction is currently empty but the LowerSYCLMetaData Pass will fill
  /// it with the required IR.
  template<typename Ptr>
#if defined(__SYCL_XILINX_HW_EMU_MODE__) || defined(__SYCL_XILINX_HW_MODE__)
  __SYCL_DEVICE_ANNOTATE("xilinx_partition_array")
#endif
  __SYCL_ALWAYS_INLINE
  inline void xilinx_partition_array(Ptr, int, int, int) {}

  /** Represent a cyclic partition.

      The single array would be partitioned into several small physical
      memories in this case. These small physical memories can be
      accessed simultaneously which drive the performance. Each
      element in the array would be partitioned to each memory in order
      and cyclically.

      That is if we have a 4-element array which contains 4 integers
      0, 1, 2, and 3. If we set factor to 2, and partition
      dimension to 1 for this cyclic partition array. Then, the
      contents of this array will be distributed to 2 physical
      memories: one contains 1, 3 and the other contains 2,4.

      \param PhyMemNum is the number of physical memories that user wants to
      have.

      \param PDim is the dimension that user wants to apply cyclic partition on.
      If PDim is 0, all dimensions will be partitioned with cyclic order.

      TODO: Deal with multi-dimension array. Now, since we can only deal with
      1-dim, PDim is set to 1 by default.
  */
  template <std::size_t PhyMemNum = 1, std::size_t PDim = 1>
  struct cyclic {
    static constexpr auto physical_mem_num = PhyMemNum;
    static constexpr auto partition_dim = PDim;
    static constexpr auto partition_type = type::cyclic;
  };


  /** Represent a block partition.

      The single array would be partitioned into several small
      physical memories and can be accessed simultaneously, too.
      However, the first physical memory will be filled up first, then
      the next.

      That is if we have a 4-element array which contains 4 integers
      0, 1, 2, and 3. If we set factor to 2, and partition
      dimension to 1 for this cyclic partition array. Then, the
      contents of this array will be distributed to 2 physical
      memories: one contains 1, 2 and the other contains 3,4.

      \param ElmInEachPhyMem is the number of elements in each physical memory
      that user wants to have.

      \param PDim is the dimension that user wants to apply block partition on.
      If PDim is 0, all dimensions will be partitioned with block order.

      TODO: Deal with multi-dimension array. Now, since we can only deal with
      1-dim, PDim is set to 1 by default.
  */
  template <std::size_t ElmInEachPhyMem = 1, std::size_t PDim = 1>
  struct block {
    static constexpr auto ele_in_each_physical_mem = ElmInEachPhyMem;
    static constexpr auto partition_dim = PDim;
    static constexpr auto partition_type = type::block;
  };


  /** Represent a complete partition.

      The single array would be partitioned into individual elements.
      That is if we have a 4-element array with one dimension, the
      array is completely partitioned into distributed RAM or 4
      independent registers.

      \param PDim is the dimension that user wants to apply complete partition
      on. If PDim is 0, all dimensions will be completely partitioned.

      TODO: Deal with multi-dimension array. Now, since we can only deal with
      1-dim, PDim is set to 1 by default.
  */
  template <std::size_t PDim = 1>
  struct complete {
    static constexpr auto partition_dim = PDim;
    static constexpr auto partition_type = type::complete;
  };


  /** Represent a none partition.

      The single array would be the same as std::array.
  */
  struct none {
    static constexpr auto partition_type = type::none;
  };
}


/** Define an array class with partition feature.

    Since on FPGA, users can customize the memory architecture in the system
    and within the CU. Array partition help us to partition single array to
    multiple memories that can be accessed simultaneously getting higher memory
    bandwidth.

    \param ValueType is the type of element.

    \param Size is the size of the array.

    \param PartitionType is the array partition type: cyclic, block, and
    complete. The default type is none.

    TODO: Deal with multi-dimension array.
*/
template <typename ValueType,
          std::size_t Size,
          typename PartitionType = partition::none>
struct partition_array {
  /** Store the array elements.

      Note that it means default initialization for partition_array
      elements, which is lazily convenient for heterogeneous
      computing */
  ValueType elems[Size];
  /// The number of elements of the 1-D array
  static constexpr auto array_size = Size;
  /// The kind of partitioning
  static constexpr auto partition_type = PartitionType::partition_type;
  /// Type of array elements
  using element_type = ValueType;
  using iterator = ValueType*;
  using const_iterator = const ValueType*;

  /// Provide iterator
  iterator begin() { return elems; }
  const_iterator begin() const { return elems; }
  iterator end() { return elems+Size; }
  const_iterator end() const { return elems+Size; }


  /// Evaluate size
  constexpr std::size_t size() const noexcept {
    return Size;
  }

  /// Construct an array
  partition_array() {
    // Add the intrinsic according expressing to the target compiler the
    // partitioning to use
    if constexpr (partition_type == partition::type::cyclic)
      partition::xilinx_partition_array(
          (ValueType __SYCL_DEVICE_ADDRSPACE(0 /*stack*/)(*)[Size])(&elems) &
              elems,
          partition_type, PartitionType::physical_mem_num,
          PartitionType::partition_dim);
    if constexpr (partition_type == partition::type::block)
      partition::xilinx_partition_array(
          (ValueType __SYCL_DEVICE_ADDRSPACE(0 /*stack*/)(*)[Size])(&elems) &
              elems,
          partition_type, PartitionType::ele_in_each_physical_mem,
          PartitionType::partition_dim);
    if constexpr (partition_type == partition::type::complete)
      partition::xilinx_partition_array(
          (ValueType __SYCL_DEVICE_ADDRSPACE(0 /*stack*/)(*)[Size])(&elems),
          partition_type, 0, PartitionType::partition_dim);
  }

  /// A constructor from some container
  template <typename SomeContainer>
  partition_array(const SomeContainer &src)
    : partition_array { } {
    /// TODO: Find a way to specialize this with a safer
    /// implementation when the size of src is at least constexpr
    std::copy_n(std::begin(src), Size, begin());
  }


  /// Construct an array from initializer_list
  template <typename SourceBasicType,
            // Only use this constructor for a real element-oriented
            // initializer_list we can assign, not for the case
            // partition_array<> a = { some_other_array }
            typename = sycl::detail::enable_if_t<std::is_convertible<SourceBasicType,
                                                            ValueType>::value>>
  constexpr partition_array(std::initializer_list<SourceBasicType> l)
    : partition_array { } {
    /// TODO: Find a way to specialize this with a safer
    /// implementation when the size of src is at least constexpr
    /// This does not work...
    /// static_assert(l.size() == Size);
    std::copy_n(std::begin(l), Size, begin());
  }


  /// Provide a subscript operator
  ValueType& operator[](std::size_t i) {
    return elems[i];
  }


  constexpr const ValueType& operator[](std::size_t i) const {
    return elems[i];
  }


  /// Return the partition type of the array
  constexpr auto get_partition_type() const -> decltype(partition_type) {
    return partition_type;
  }
};

} // namespace cl::sycl::ext::xilinx

}

#endif// SYCL_XILINX_FPGA_PARTITION_ARRAY_HPP
#endif
