#ifndef BASEFLASHHYPOTHESIS_CXX
#define BASEFLASHHYPOTHESIS_CXX

#include "BaseFlashHypothesis.h"
#include "OpT0FinderException.h"
namespace flashmatch {

  BaseFlashHypothesis::BaseFlashHypothesis(std::string name)
    : BaseAlgorithm(kFlashHypothesis,name)
    , _channel_mask(DetectorSpecs::GetME().NOpDets(),true)
    , _uncoated_pmt_list(DetectorSpecs::GetME().NOpDets(),false)
  {}

  void BaseFlashHypothesis::_Configure_(const Config_t &pset)
  {
    _threshold_proximity = pset.get<double>("ProximityThreshold",5.0);
    _segment_size = pset.get<double>("SegmentSize",0.5);

    _global_qe = pset.get<double>("GlobalQE");
    _global_qe_refl = pset.get<double>("GlobalQERefl", -1);

    // This is meant to be a list of channel ids to use
    _chs_to_use = pset.get<std::vector<int> >("ChannelToUse",_chs_to_use);
    if(!_chs_to_use.empty()) {
      this->SetChannelMask(_chs_to_use); // This will set the channel mask based on the channel ids
    }
    else { //assume all channels are used
      for (size_t i = 0; i < _channel_mask.size(); i++) {
        _channel_mask[i] = true;
      }
    }

    _qe_v.clear();
    _qe_v = pset.get<std::vector<double> >("CCVCorrection",_qe_v);
    if(_qe_v.empty()) _qe_v.resize(DetectorSpecs::GetME().NOpDets(),1.0);
    if(_qe_v.size() != DetectorSpecs::GetME().NOpDets()) {
      FLASH_CRITICAL() << "CCVCorrection factor array has size " << _qe_v.size()
      << " != number of opdet (" << DetectorSpecs::GetME().NOpDets() << ")!" << std::endl;
      throw OpT0FinderException();
    }

    //Debug statements
    FLASH_DEBUG() << "Channel mask size: " << _channel_mask.size() << std::endl;
    FLASH_DEBUG() << "fNOpDets: " << DetectorSpecs::GetME().NOpDets() << std::endl;
    for(size_t i=0; i<_channel_mask.size(); ++i) {
      FLASH_DEBUG() << "Channel " << i << " is masked to " << _channel_mask[i] << std::endl;
    }

  }

  //TODO: Implement this mask for a given list of channel ids or conditions
  void BaseFlashHypothesis::InitializeMask(Flash_t &flash) const{
    flash.pds_mask_v.clear();
    flash.pds_mask_v.resize(DetectorSpecs::GetME().NOpDets(), 0); // 1 means skip
  }

  Flash_t BaseFlashHypothesis::GetEstimate(const QCluster_t& tpc) const
  {
    Flash_t res;
    //res.pe_v.resize(OpDetXArray().size());

    FillEstimate(tpc,res);
    return res;
  }

}
#endif
