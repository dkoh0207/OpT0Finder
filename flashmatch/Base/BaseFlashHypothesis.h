/**
 * \file FlashHypothesis.h
 *
 * \ingroup Base
 * 
 * \brief Class def header for a class FlashHypothesis
 *
 * @author kazuhiro
 */

/** \addtogroup Base

    @{*/
#ifndef BASEFLASHHYPOTHESIS_H
#define BASEFLASHHYPOTHESIS_H

#include "BaseAlgorithm.h"

namespace flashmatch {
  /**
     \class FlashHypothesis
     User defined class FlashHypothesis ... these comments are used to generate
     doxygen documentation!
  */
  class BaseFlashHypothesis : public flashmatch::BaseAlgorithm {
    
  public:

    /// Default constructor
    BaseFlashHypothesis(const std::string name="noname");
    
    /// Default destructor
    ~BaseFlashHypothesis(){}

    // _Configure_
    void _Configure_(const Config_t &pset);

    /// Method to create flashmatch::Flash_t object and return
    Flash_t GetEstimate(const QCluster_t&) const;

    /// Method to simply fill provided reference of flashmatch::Flash_t
    virtual void FillEstimate(const QCluster_t&, Flash_t&) const = 0;

    /// Sets pds_mask_v in flash method
    void InitializeMask(Flash_t &flash) const;

    /// Sets the channels to use
    void SetChannelMask(std::vector<int> ch_to_use) { 
      // Initialize all channels to false
      FLASH_DEBUG() << "ch_to_use size: " << ch_to_use.size() << std::endl;
      for (size_t i = 0; i < _channel_mask.size(); i++) {
        _channel_mask[i] = false;
      }
      // Set the channels to use to true
      for (size_t i = 0; i < ch_to_use.size(); i++) {
        FLASH_DEBUG() << "Setting channel mask: " << ch_to_use[i] << " to true" << std::endl;
        _channel_mask[ch_to_use[i]] = true;
      }
    }

    /// Sets the channel type (pmt vs. xarapuca)
    void SetChannelType(std::vector<int> ch_type) { _channel_type = ch_type; }

    /// Sets the channels sensitive to visible light
    void SetUncoatedPMTs(std::vector<int> ch_uncoated) { _uncoated_pmt_list = ch_uncoated; }

  protected:

    std::vector<int> _channel_type; 
    std::vector<int> _chs_to_use; ///< The list of channels mask use of
    std::vector<bool> _channel_mask; ///< The list of channels mask use of
    std::vector<int> _uncoated_pmt_list; ///< A list of opdet sensitive to visible (reflected) light
    double _threshold_proximity; ///< Threshold for proximity
    double _segment_size; ///< Segment size
    double _global_qe;
    std::vector<double> _qe_v;     ///< OpDet-wise relative QE for direct light
    double _global_qe_refl;        ///< Global QE for reflected light
  };
}
#endif
/** @} */ // end of doxygen group 

