#pragma once
#include <deal.II/base/tensor.h>

// template <typename ComplexType, typename Real, int rank, int dim>
// dealii::Tensor<rank, dim, ComplexType>
// to_complex(const dealii::Tensor<rank, dim, Real> &real,
//            const dealii::Tensor<rank, dim, Real> &imag)
//{
//   ComplexType complex_tensor;
//   for (unsigned int i = 0; i < dim; ++i)
//     {
//       constexpr if (rank == 1) { complex_tensor[i] = ComplexType(tensor[i]);
//       } else { complex_tensor[i] = to_complex<Real>(real[i], imag[i]); }
//     }
//   return complex_tensor;
// }

template <typename ComplexType, typename Real, int rank, int dim>
dealii::Tensor<rank, dim, ComplexType>
to_complex(const dealii::Tensor<rank, dim, Real> &real,
           const dealii::Tensor<rank, dim, Real> &imag)
{
  dealii::Tensor<rank, dim, ComplexType> complex_tensor;
  for (unsigned int i = 0;
       i < dealii::Tensor<rank, dim>::n_independent_components; ++i)
    {
      dealii::TableIndices<rank> index =
          dealii::Tensor<rank, dim, Real>::unrolled_to_component_indices(i);
      complex_tensor[index] = ComplexType(real[index], imag[index]);
    }
  return complex_tensor;
}

/* template <template <typename> typename ComplexTempl, typename Real, int rank,
          int dim>
auto
to_complex(const dealii::Tensor<rank, dim, Real> &real,
           const dealii::Tensor<rank, dim, Real> &imag)
    -> dealii::Tensor<rank, dim, ComplexTempl<Real>>; */

template <typename Real, typename ComplexType, int rank, int dim>
std::pair<dealii::Tensor<rank, dim, Real>, dealii::Tensor<rank, dim, Real>>
from_complex(const dealii::Tensor<rank, dim, ComplexType> &complex_tensor)
{
  std::pair<dealii::Tensor<rank, dim, Real>, dealii::Tensor<rank, dim, Real>>
      tensor_pair;
  for (unsigned int i = 0;
       i < dealii::Tensor<rank, dim>::n_independent_components; ++i)
    {
      dealii::TableIndices<rank> index =
          dealii::Tensor<rank, dim, Real>::unrolled_to_component_indices(i);
      tensor_pair.first[index]  = complex_tensor[index].real();
      tensor_pair.second[index] = complex_tensor[index].imag();
    }
  return tensor_pair;
}

/* template <typename ComplexType, int rank, int dim,
          typename Real = decltype(ComplexType().real())>
auto
from_complex(const dealii::Tensor<rank, dim, ComplexType> &complex_tensor)
    -> std::pair<dealii::Tensor<rank, dim, Real>,
                 dealii::Tensor<rank, dim, Real>>; */

template <template <typename> typename InArrayType,
          template <typename> typename OutArrayType, typename T, int rank,
          int dim>
dealii::Tensor<rank, dim, OutArrayType<T>>
to_tensor(const InArrayType<dealii::Tensor<rank, dim, T>> &input)
{
  dealii::Tensor<rank, dim, OutArrayType<T>> output;
  for (unsigned int i = 0;
       i < dealii::Tensor<rank, dim>::n_independent_components; ++i)
    {
      dealii::TableIndices<rank> index =
          dealii::Tensor<rank, dim, T>::unrolled_to_component_indices(i);
      for (unsigned int j = 0;
           j < InArrayType<dealii::Tensor<rank, dim, T>>::size(); ++j)
        {
          output[index][j] = input[j][index];
        }
    }
  return output;
}