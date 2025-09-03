#define __STDCPP_WANT_MATH_SPEC_FUNCS__ 1

#include "SemiAnalyticalModel.h"
#include "PhotonPropagationUtils.h"

#include "TMath.h"

#include <cmath>
#include <iostream>
#include <vector>
#include <cassert>

#if USING_LARSOFT == 0
namespace flashmatch {
  /// Configuration object
  using Config_t = flashmatch::FMParams;
}
#else
namespace flashmatch {
  /// Configuration object
  using Config_t = fhicl::ParameterSet;
}
#endif

#if USING_LARSOFT == 0
#include <omp.h>
#define NUM_THREADS 1
#endif


namespace flashmatch{
    static SemiAnalyticalModelFactory __global_SemiAnalyticalModelFactory__;

    SemiAnalyticalModel::SemiAnalyticalModel(const std::string name)
    : BaseFlashHypothesis(name)
    {
        #if USING_LARSOFT == 0
        omp_set_num_threads(NUM_THREADS); 
        #endif
    }
  // constructor
  void SemiAnalyticalModel::_Configure_(const Config_t& pset)
  {
    // Call the base class's _Configure_ method
    BaseFlashHypothesis::_Configure_(pset);
    //Geometry specs
    _qe_refl_v = pset.get<std::vector<double> >("VISEfficiency",_qe_refl_v);
    if(_qe_refl_v.empty()) _qe_refl_v.resize(DetectorSpecs::GetME().NOpDets(),1.0);
    if(_qe_refl_v.size() != DetectorSpecs::GetME().NOpDets()) {
      FLASH_CRITICAL() << "VIS Efficiency factor array has size " << _qe_refl_v.size()
                       << " != number of opdet (" << DetectorSpecs::GetME().NOpDets() << ")!" << std::endl;
      throw OpT0FinderException();
    }

    fNTPC = DetectorSpecs::GetME().NTPCs();
    fActiveVolume = DetectorSpecs::GetME().ActiveVolume();
    fanode_centre = TVector3(fActiveVolume.Min()[0], //Set to minx until we find a way to chose the tpc. The x-coord is not used.
                    fActiveVolume.Center()[1],
                    fActiveVolume.Center()[2]);
    fcathode_centre = TVector3(fActiveVolume.Center()[0], //Set to center until we find a way to chose the tpc. The x-coord is not used.
                    fActiveVolume.Center()[1],
                    fActiveVolume.Center()[2]);
    fDoReflectedLight = pset.get<bool>("DoReflectedLight",true);
    fIncludeAnodeReflections = pset.get<bool>("includeAnodeReflections",false);
    std::vector<double> _abs_length_spectrum = pset.get<std::vector<double>>("AbsLengthSpectrum");
    std::vector<double> _abs_length_energy = pset.get<std::vector<double>>("AbsLengthEnergies");
    // Create a map from absorption spectrum and energies
    for (size_t i = 0; i < _abs_length_spectrum.size(); ++i) {
      abs_length_spectrum[_abs_length_energy[i]] = _abs_length_spectrum[i];
    }
    fUseXeAbsorption = pset.get<bool>("useXeAbsorption",false);
    fspherical_type = DetectorSpecs::GetME().GetSphericalType();
    fspherical_orientation = DetectorSpecs::GetME().GetSphericalOrientation();
    fspherical_ids = DetectorSpecs::GetME().GetSphericalIds();
    
    frectengular_type = DetectorSpecs::GetME().GetRectengularType();
    frectengular_orientation = DetectorSpecs::GetME().GetRectengularOrientation();
    frectengular_height = DetectorSpecs::GetME().GetRectengularHeight();
    frectengular_width = DetectorSpecs::GetME().GetRectengularWidth();
    frectengular_ids = DetectorSpecs::GetME().GetRectengularIds();
    fOpDetector = opticalDetectors();
    fNOpDets = fOpDetector.size();
    assert(fNOpDets == DetectorSpecs::GetME().NOpDets());

    fIsFlatPDCorr = pset.get<bool>("FlatPDCorr", false);
    fIsFlatPDCorrLat = pset.get<bool>("FlatPDCorrLat", false);
    fIsDomePDCorr = pset.get<bool>("DomePDCorr", false);
    fdelta_angulo_vuv = pset.get<double>("delta_angulo_vuv", 10);
    fradius = pset.get<double>("PMT_radius", 10.16);
    fApplyFieldCageTransparency = pset.get<bool>("ApplyFieldCageTransparency", false);
    fFieldCageTransparencyLateral = pset.get<double>("FieldCageTransparencyLateral", 1.0);
    fFieldCageTransparencyCathode = pset.get<double>("FieldCageTransparencyCathode", 1.0);

    if (!fIsFlatPDCorr && !fIsDomePDCorr && !fIsFlatPDCorrLat) {
      FLASH_CRITICAL()
        << "Both isFlatPDCorr/isFlatPDCorrLat and isDomePDCorr parameters are "
            "false, at least one type of parameterisation is required for the "
            "semi-analytic light simulation."
        << "\n";
        throw OpT0FinderException();
    }
    if (fIsFlatPDCorr) {
      fGHvuvpars_flat = pset.get<std::vector<std::vector<double>>>("GH_PARS_flat");
      fborder_corr_angulo_flat = pset.get<std::vector<double>>("GH_border_angulo_flat");
      fborder_corr_flat = pset.get<std::vector<std::vector<double>>>("GH_border_flat");
    }
    if (fIsFlatPDCorrLat) {
      fGHvuvpars_flat_lateral =
        pset.get<std::vector<std::vector<double>>>("GH_PARS_flat_lateral");
      fborder_corr_angulo_flat_lateral =
        pset.get<std::vector<double>>("GH_border_angulo_flat_lateral");
      fborder_corr_flat_lateral =
        pset.get<std::vector<std::vector<double>>>("GH_border_flat_lateral");
    }
    if (fIsDomePDCorr) {
      fGHvuvpars_dome = pset.get<std::vector<std::vector<double>>>("GH_PARS_dome");
      fborder_corr_angulo_dome = pset.get<std::vector<double>>("GH_border_angulo_dome");
      fborder_corr_dome = pset.get<std::vector<std::vector<double>>>("GH_border_dome");
    }

    // Load corrections for VIS semi-analytic hits
    if (fDoReflectedLight) {
      fdelta_angulo_vis = pset.get<double>("delta_angulo_vis");
      if (fIsFlatPDCorr) {
        fvis_distances_x_flat = pset.get<std::vector<double>>("VIS_distances_x_flat");
        fvis_distances_r_flat = pset.get<std::vector<double>>("VIS_distances_r_flat");
        fvispars_flat =
          pset.get<std::vector<std::vector<std::vector<double>>>>("VIS_correction_flat");
      }
      if (fIsDomePDCorr) {
        fvis_distances_x_dome = pset.get<std::vector<double>>("VIS_distances_x_dome");
        fvis_distances_r_dome = pset.get<std::vector<double>>("VIS_distances_r_dome");
        fvispars_dome =
          pset.get<std::vector<std::vector<std::vector<double>>>>("VIS_correction_dome");
      }

      // set cathode plane struct for solid angle function
      fcathode_plane.h = fActiveVolume.SizeY();
      fcathode_plane.w = fActiveVolume.SizeZ();
      fplane_depth = std::abs(fcathode_centre[0]);
    }

    // Load corrections for Anode reflections configuration
    if (fIncludeAnodeReflections) {
      fdelta_angulo_vis = pset.get<double>("delta_angulo_vis");
      fAnodeReflectivity = pset.get<double>("AnodeReflectivity");

      if (fIsFlatPDCorr) {
        fvis_distances_x_flat = pset.get<std::vector<double>>("VIS_distances_x_flat");
        fvis_distances_r_flat = pset.get<std::vector<double>>("VIS_distances_r_flat");
        fvispars_flat =
          pset.get<std::vector<std::vector<std::vector<double>>>>("VIS_correction_flat");
      }

      if (fIsFlatPDCorrLat) {
        fvis_distances_x_flat_lateral =
          pset.get<std::vector<double>>("VIS_distances_x_flat_lateral");
        fvis_distances_r_flat_lateral =
          pset.get<std::vector<double>>("VIS_distances_r_flat_lateral");
        fvispars_flat_lateral = pset.get<std::vector<std::vector<std::vector<double>>>>(
          "VIS_correction_flat_lateral");
      }

      // set anode plane struct for solid angle function
      fanode_plane.h = fActiveVolume.SizeY();
      fanode_plane.w = fActiveVolume.SizeZ();
      fanode_plane_depth = fanode_centre[0];

    }

    // set absorption length
    fvuv_absorption_length = VUVAbsorptionLength();

    FLASH_INFO() << "Semi-analytical model initialized." << std::endl;

    //Debug statements for semi-analytical model
    //Print();
    //printVISParameters();
    //printVUVParameters();
  }

  void SemiAnalyticalModel::FillEstimate(const QCluster_t& trk, Flash_t &flash) const
    {
    if(flash.pe_v.empty()) flash.pe_v.resize(fNOpDets);
    if(flash.pe_err_v.empty()) flash.pe_err_v.resize(fNOpDets);
    if(flash.pe_true_v.empty()) flash.pe_true_v.resize(fNOpDets);
    if(flash.closest_pds_v.empty()) flash.closest_pds_v.resize(fNOpDets);

    assert(flash.pe_v.size()     == fNOpDets);
    assert(flash.pe_true_v.size() == fNOpDets);
    assert(flash.pe_err_v.size() == fNOpDets);
    assert(flash.closest_pds_v.size() == fNOpDets);
    for (auto& v : flash.pe_v      ) {v = 0;}
    for (auto& v : flash.pe_err_v  ) {v = 0;}
    for (auto& v : flash.pe_true_v ) {v = 0;}
    for (auto& v : flash.closest_pds_v) {v = 1.e9;}
    // Temporary for printout
    double q_tot = 0;
    // end of temporary for printout
    for ( size_t ipt = 0; ipt < trk.size(); ++ipt) {
      /// Get the 3D point in space from where photons should be propagated
      auto const& pt = trk[ipt];

      geoalgo::Point_t const xyz = {pt.x, pt.y, pt.z};

      double n_original_photons = pt.q;
      q_tot += n_original_photons;

      std::vector<double> direct_visibilities;
      detectedDirectVisibilities(direct_visibilities, xyz);


      std::vector<double> reflected_visibilities;
      detectedReflectedVisibilities(reflected_visibilities, xyz);

      // if (ipt < 3){
      //   for (size_t op_det=0; op_det<direct_visibilities.size(); ++op_det) {
      //     if (op_det > 40){
      //       break;
      //     }
      //     std::cout<<"[x,y,z]: "<<pt.x<<", "<<pt.y<<", "<<pt.z<<std::endl;
      //     std::cout<<"op_det: "<<op_det<<std::endl;
      //     std::cout<<"direct_visibilities[op_det]: "<<direct_visibilities[op_det]<<std::endl;
      //     std::cout<<"reflected_visibilities[op_det]: "<<reflected_visibilities[op_det]<<std::endl;
      //   }
      // }

      //
      // Fill Estimate with Direct light
      //
      for (size_t op_det=0; op_det<direct_visibilities.size(); ++op_det) {
        if (_channel_mask[op_det] == false) {
          continue;
        }
        const double visibility = direct_visibilities[op_det];

        double q = n_original_photons * visibility * _global_qe * _qe_v[op_det];

        flash.pe_v[op_det] += q;
        geoalgo::Point_t const op_det_pos = fOpDetector[op_det].center;
        // Update the closest distance to the PMT - only need to do this with direct or indirect light, add the visibility check if the charge is in a separate TPC.
        double distance = (xyz - op_det_pos).Length();
        if (distance < flash.closest_pds_v[op_det] && visibility > 0) {
          FLASH_DEBUG() << "Updating closest distance to PMT " << op_det << " to " << distance << std::endl;
          flash.closest_pds_v[op_det] = distance;
        }
      }

      //
      // Fill Estimate with Reflected light
      //
      for (size_t op_det=0; op_det<reflected_visibilities.size(); ++op_det) {
        if (_channel_mask[op_det] == false) {
          continue;
        }
        const double visibility = reflected_visibilities[op_det];
        double q = n_original_photons * visibility * _global_qe_refl * _qe_refl_v[op_det];

        flash.pe_v[op_det] += q;
      }
    }
    //Count number of valid channels
    // int cnt = 0;
    // std::cout<<"flash.pe_v.size(): "<<flash.pe_v.size()<<std::endl;
    // for (size_t op_det=0; op_det<flash.pe_v.size(); ++op_det) {
    //   if (flash.pe_v[op_det] > 0){
    //     cnt++;
    //   }
    //   else{
    //     continue;
    //   }
    //   std::cout<<"op_det: "<<op_det<<std::endl;
    //   std::cout<<"flash.pe_v[op_det]: "<<flash.pe_v[op_det]<<std::endl;
    //   //if (cnt > 8){
    //   if (op_det > 40){
    //     break;
    //   }
    // }
    
    // Print outs to check validity of filling the flashes
    FLASH_INFO() << "Filled flash with " << _channel_mask.size() << " PMTs ... " 
    << " Valid: " << flash.Valid(fNOpDets) << " Total PE: " << flash.TotalPE() 
    << " trk.size(): " << trk.size()
    << "Flash [x,y,z] -> [Total PE] : [" << flash.x << ", " << flash.y << ", " << flash.z << "] -> [" << flash.TotalPE() << "]"
    << " q_total: " << q_tot
    << " closest_pds_v (first 10): " << flash.closest_pds_v[0] << ", " << flash.closest_pds_v[1] << ", " << flash.closest_pds_v[2] << ", " << flash.closest_pds_v[3] << ", " << flash.closest_pds_v[4] << ", " << flash.closest_pds_v[5] << ", " << flash.closest_pds_v[6] << ", " << flash.closest_pds_v[7] << ", " << flash.closest_pds_v[8] << ", " << flash.closest_pds_v[9]
    << " idx: " << flash.idx << std::endl;
    // Check validity of flash
    if (!flash.Valid(fNOpDets)) {
      FLASH_CRITICAL() << "Flash is not valid. Check the flash filling." 
      << " fNOpDets" << fNOpDets
      << " flash.pe_v.size() " << flash.pe_v.size() 
      << " flash.pe_err_v.size() " << flash.pe_err_v.size()
      << " flash.idx " << flash.idx << std::endl;
      throw OpT0FinderException();
    }
    for (size_t op_det=0; op_det<flash.pe_v.size(); ++op_det) {
      if (flash.pe_v[op_det] > 0) {
        FLASH_DEBUG()<<"op_det: "<<op_det<<" pe_v: "<<flash.pe_v[op_det]<<std::endl;
      }
    }

  }

  double SemiAnalyticalModel::VUVAbsorptionLength() const
  {
    // determine LAr absorption length in cm
    std::vector<double> x_v, y_v;
    for (auto elem : abs_length_spectrum) {
      x_v.push_back(elem.first);
      y_v.push_back(elem.second);
    }
    double vuv_absorption_length;
    if (fUseXeAbsorption)
      vuv_absorption_length =
        interpolate(x_v, y_v, 7.1, false); // 7.1 eV: peak of Xe VUV emission spectrum
    else
      vuv_absorption_length =
        interpolate(x_v, y_v, 9.7, false); // 9.7 eV: peak of Ar VUV emission spectrum
    if (vuv_absorption_length <= 0) {
      FLASH_CRITICAL()
        << "Error: VUV Absorption Length is 0 or negative.\n";
      throw OpT0FinderException();
    }
    return vuv_absorption_length;
  }

  //......................................................................
  // VUV semi-analytical model visibility calculation
  void SemiAnalyticalModel::detectedDirectVisibilities(std::vector<double>& DetectedVisibilities,
                                                        geoalgo::Point_t const& ScintPoint) const
  {

    DetectedVisibilities.resize(fNOpDets);
    for (size_t OpDet = 0; OpDet < fNOpDets; ++OpDet) {
      if (!isOpDetInSameTPC(ScintPoint, fOpDetector[OpDet].center)) {
        DetectedVisibilities[OpDet] = 0.;
        continue;
      }

      DetectedVisibilities[OpDet] = VUVVisibility(ScintPoint, fOpDetector[OpDet]);
    }
  }

  double SemiAnalyticalModel::VUVVisibility(geoalgo::Point_t const& ScintPoint,
                                            OpticalDetector const& opDet) const
  {
    // distance and angle between ScintPoint and OpDet center
    geoalgo::Vector_t const relative = ScintPoint - opDet.center;
    const double distance = relative.Length();
    double cosine;
    if (opDet.orientation == 2)
      cosine = std::abs(relative[2]) / distance;
    else if (opDet.orientation == 1)
      cosine = std::abs(relative[1]) / distance;
    else
      cosine = std::abs(relative[0]) / distance;
    const double theta = fast_acos(cosine) * 180. / geoalgo::kPI;

    double solid_angle = 0.;
    // ARAPUCAS/Bars (rectangle)
    if (opDet.type == 0) {
      // get scintillation point coordinates relative to arapuca window centre
      geoalgo::Vector_t const abs_relative{
        std::abs(relative[0]), std::abs(relative[1]), std::abs(relative[2])};
      solid_angle = Rectangle_SolidAngle(Dims{opDet.h, opDet.w}, abs_relative, opDet.orientation);
    }
    // PMTs (dome)
    else if (opDet.type == 1) {
      solid_angle = Omega_Dome_Model(distance, theta);
    }
    // PMTs (disk)
    else if (opDet.type == 2) {
      const double zy_offset = std::sqrt(relative[1] * relative[1] + relative[2] * relative[2]);
      const double x_distance = std::abs(relative[0]);
      solid_angle = Disk_SolidAngle(zy_offset, x_distance, fradius);
    }
    else {
      FLASH_CRITICAL()
        << "Error: Invalid optical detector shape requested - configuration "
            "error in semi-analytical model, only rectangular, dome or disk "
            "optical detectors are supported."
        << "\n";
       throw OpT0FinderException(); 
    }

    // calculate visibility by geometric acceptance
    // accounting for solid angle and LAr absorbtion length
    double visibility_geo =
      std::exp(-1. * distance / fvuv_absorption_length) * (solid_angle / (4 * geoalgo::kPI));
    // apply Gaisser-Hillas correction for Rayleigh scattering distance
    // and angular dependence offset angle bin
    const size_t j = (theta / fdelta_angulo_vuv);

    // determine GH parameters, accounting for border effects
    // radial distance from centre of detector (Y-Z standard / X-Z laterals)
    double r = 0;
    if (opDet.orientation == 2)
      r = std::hypot(ScintPoint[0] - fcathode_centre[0], ScintPoint[1] - fcathode_centre[1]);
    else if (opDet.orientation == 1)
      r = std::hypot(ScintPoint[0] - fcathode_centre[0], ScintPoint[2] - fcathode_centre[2]);
    else
      r = std::hypot(ScintPoint[1] - fcathode_centre[1], ScintPoint[2] - fcathode_centre[2]);

    double pars_ini[4] = {0, 0, 0, 0};
    double s1 = 0;
    double s2 = 0;
    double s3 = 0;
    // flat PDs
    if ((opDet.type == 0 || opDet.type == 2) && (fIsFlatPDCorr || fIsFlatPDCorrLat)) {
      if ((opDet.orientation == 1 && fIsFlatPDCorrLat) ||
          (opDet.orientation == 2 && fIsFlatPDCorrLat)) { // laterals
        pars_ini[0] = fGHvuvpars_flat_lateral[0][j];
        pars_ini[1] = fGHvuvpars_flat_lateral[1][j];
        pars_ini[2] = fGHvuvpars_flat_lateral[2][j];
        pars_ini[3] = fGHvuvpars_flat_lateral[3][j];
        s1 =
          interpolate(fborder_corr_angulo_flat_lateral, fborder_corr_flat_lateral[0], theta, true);
        s2 =
          interpolate(fborder_corr_angulo_flat_lateral, fborder_corr_flat_lateral[1], theta, true);
        s3 =
          interpolate(fborder_corr_angulo_flat_lateral, fborder_corr_flat_lateral[2], theta, true);
      }
      else if (opDet.orientation == 0 && fIsFlatPDCorr) { // cathode/anode
        pars_ini[0] = fGHvuvpars_flat[0][j];
        pars_ini[1] = fGHvuvpars_flat[1][j];
        pars_ini[2] = fGHvuvpars_flat[2][j];
        pars_ini[3] = fGHvuvpars_flat[3][j];
        s1 = interpolate(fborder_corr_angulo_flat, fborder_corr_flat[0], theta, true);
        s2 = interpolate(fborder_corr_angulo_flat, fborder_corr_flat[1], theta, true);
        s3 = interpolate(fborder_corr_angulo_flat, fborder_corr_flat[2], theta, true);
      }
      else {
        FLASH_CRITICAL()
          << "Error: flat optical detectors are found, but parameters are "
              "missing - configuration error in semi-analytical model."
          << "\n";
          throw OpT0FinderException();
      }
    }
    // dome PDs
    else if (opDet.type == 1 && fIsDomePDCorr) {
      pars_ini[0] = fGHvuvpars_dome[0][j];
      pars_ini[1] = fGHvuvpars_dome[1][j];
      pars_ini[2] = fGHvuvpars_dome[2][j];
      pars_ini[3] = fGHvuvpars_dome[3][j];
      s1 = interpolate(fborder_corr_angulo_dome, fborder_corr_dome[0], theta, true);
      s2 = interpolate(fborder_corr_angulo_dome, fborder_corr_dome[1], theta, true);
      s3 = interpolate(fborder_corr_angulo_dome, fborder_corr_dome[2], theta, true);
    }
    else {
      FLASH_CRITICAL()
        << "Error: Invalid optical detector shape requested or corrections are "
            "missing - configuration error in semi-analytical model."
        << "\n";
        throw OpT0FinderException();
    }

    // add border correction to parameters
    pars_ini[0] = pars_ini[0] + s1 * r;
    pars_ini[1] = pars_ini[1] + s2 * r;
    pars_ini[2] = pars_ini[2] + s3 * r;
    pars_ini[3] = pars_ini[3];

    // calculate correction
    double GH_correction = Gaisser_Hillas(distance, pars_ini);

    // determine corrected visibility of photo-detector
    if (std::isnan(GH_correction) || std::isnan(visibility_geo) || std::isnan(cosine)) {  
      // TODO: have solid angle not be nan
      // <<" solid_angle: "<<solid_angle << std::endl;
      // solid angle is nan sometimes
      // FLASH_DEBUG()
      //   << "NaN value in VUV visibility estimation"
      //       "parameters. Setting visibility to 0."
      //   << "\n"
      //   << "GH_correction: " << GH_correction << " visibility_geo: " << visibility_geo << " cosine: "
      //   << cosine << std::endl;
      //throw OpT0FinderException();
      return 0;
    }
    return GH_correction * visibility_geo / cosine;
  }

  //......................................................................
  // VIS semi-analytical model visibility calculation
  void SemiAnalyticalModel::detectedReflectedVisibilities(
    std::vector<double>& ReflDetectedVisibilities,
    geoalgo::Point_t const& ScintPoint,
    bool AnodeMode) const
  {
    // 1). calculate visibility of VUV photons on
    // reflective foils via solid angle + Gaisser-Hillas
    // corrections:

    // get scintpoint coords relative to centre of cathode plane and set plane
    // dimensions
    geoalgo::Vector_t ScintPoint_relative;
    Dims plane_dimensions;
    double plane_depth;
    if (AnodeMode) {
      plane_dimensions = fanode_plane;
      plane_depth = fanode_plane_depth;
      ScintPoint_relative = geoalgo::Vector_t(std::abs(ScintPoint[0] - fanode_plane_depth),
                                          std::abs(ScintPoint[1] - fanode_centre[1]),
                                          std::abs(ScintPoint[2] - fanode_centre[2]));
    }
    else {
      plane_dimensions = fcathode_plane;
      plane_depth = ScintPoint[0] < 0. ? -fplane_depth : fplane_depth;
      ScintPoint_relative = geoalgo::Vector_t(std::abs(ScintPoint[0] - plane_depth),
                                          std::abs(ScintPoint[1] - fcathode_centre[1]),
                                          std::abs(ScintPoint[2] - fcathode_centre[2]));
    }
    // calculate solid angle of anode/cathode from the scintillation point,
    // orientation always = 0 (anode/cathode)
    double solid_angle_cathode = Rectangle_SolidAngle(plane_dimensions, ScintPoint_relative, 0);

    // calculate distance and angle between ScintPoint and hotspot
    // vast majority of hits in hotspot region directly infront of scintpoint,
    // therefore consider attenuation for this distance and on axis GH instead of
    // for the centre coordinate
    double distance_cathode = std::abs(plane_depth - ScintPoint[0]);
    // calculate hits on cathode plane via geometric acceptance
    double cathode_visibility_geo = std::exp(-1. * distance_cathode / fvuv_absorption_length) *
                                    (solid_angle_cathode / (4. * geoalgo::kPI));
    // determine Gaisser-Hillas correction including border effects
    // use flat correction
    double r = std::hypot(ScintPoint[1] - fcathode_centre[1], ScintPoint[2] - fcathode_centre[2]);
    double pars_ini[4] = {0, 0, 0, 0};
    double s1 = 0;
    double s2 = 0;
    double s3 = 0;
    if (fIsFlatPDCorr) {
      pars_ini[0] = fGHvuvpars_flat[0][0];
      pars_ini[1] = fGHvuvpars_flat[1][0];
      pars_ini[2] = fGHvuvpars_flat[2][0];
      pars_ini[3] = fGHvuvpars_flat[3][0];
      s1 = interpolate(fborder_corr_angulo_flat, fborder_corr_flat[0], 0, true);
      s2 = interpolate(fborder_corr_angulo_flat, fborder_corr_flat[1], 0, true);
      s3 = interpolate(fborder_corr_angulo_flat, fborder_corr_flat[2], 0, true);
    }
    else {
      FLASH_CRITICAL()
        << "Error: flat optical detector VUV correction required for reflected "
            "semi-analytic hits. - configuration error in semi-analytical model."
        << "\n";
        throw OpT0FinderException();
    }

    // add border correction
    pars_ini[0] = pars_ini[0] + s1 * r;
    pars_ini[1] = pars_ini[1] + s2 * r;
    pars_ini[2] = pars_ini[2] + s3 * r;
    pars_ini[3] = pars_ini[3];


    // calculate corrected number of hits
    double GH_correction = Gaisser_Hillas(distance_cathode, pars_ini);
    const double cathode_visibility_rec = GH_correction * cathode_visibility_geo;

    // 2). detemine visibility of each PD
    const geoalgo::Point_t hotspot = {plane_depth, ScintPoint[1], ScintPoint[2]};
    ReflDetectedVisibilities.resize(fNOpDets);
    for (size_t OpDet = 0; OpDet < fNOpDets; ++OpDet) {
      if (!isOpDetInSameTPC(ScintPoint, fOpDetector[OpDet].center)) {
        ReflDetectedVisibilities[OpDet] = 0.;
        continue;
      }

      ReflDetectedVisibilities[OpDet] =
        VISVisibility(ScintPoint, fOpDetector[OpDet], cathode_visibility_rec, hotspot, AnodeMode);
    }
  }

  double SemiAnalyticalModel::VISVisibility(geoalgo::Point_t const& ScintPoint,
                                            OpticalDetector const& opDet,
                                            const double cathode_visibility,
                                            geoalgo::Point_t const& hotspot,
                                            bool AnodeMode) const
  {
    // set correct plane_depth
    double plane_depth;
    if (AnodeMode)
      plane_depth = fanode_plane_depth;
    else
      plane_depth = ScintPoint[0] < 0. ? -fplane_depth : fplane_depth;

    // calculate visibility of the optical
    // detector from the hotspot using solid angle:

    geoalgo::Vector_t const emission_relative = hotspot - opDet.center;

    // calculate distances and angles for application of corrections
    // distance from hotspot to optical detector
    const double distance_vis = emission_relative.Length();
    //  angle between hotspot and optical detector
    double cosine_vis;
    if (opDet.orientation == 2) { // lateral fixed at z
      cosine_vis = std::abs(emission_relative[2]) / distance_vis;
    }
    else if (opDet.orientation == 1) { // lateral fixed at y
      cosine_vis = std::abs(emission_relative[1]) / distance_vis;
    }
    else { // anode/cathode (default)
      cosine_vis = std::abs(emission_relative[0]) / distance_vis;
    }
    const double theta_vis = fast_acos(cosine_vis) * 180. / geoalgo::kPI;
    // calculate solid angle of optical channel
    double solid_angle_detector = 0.;
    // ARAPUCAS/Bars (rectangle)
    if (opDet.type == 0) {
      // get hotspot coordinates relative to opDet
      geoalgo::Vector_t const abs_emission_relative{std::abs(emission_relative[0]),
                                                std::abs(emission_relative[1]),
                                                std::abs(emission_relative[2])};
      solid_angle_detector =
        Rectangle_SolidAngle(Dims{opDet.h, opDet.w}, abs_emission_relative, opDet.orientation);
    }
    // PMTS (dome)
    else if (opDet.type == 1) {
      solid_angle_detector = Omega_Dome_Model(distance_vis, theta_vis);
    }
    // PMTs (disk)
    else if (opDet.type == 2) {
      const double zy_offset = std::sqrt(emission_relative[1] * emission_relative[1] +
                                          emission_relative[2] * emission_relative[2]);
      const double x_distance = std::abs(emission_relative[0]);
      solid_angle_detector = Disk_SolidAngle(zy_offset, x_distance, fradius);
    }
    else {
      FLASH_CRITICAL()
        << "Error: Invalid optical detector shape requested - configuration "
            "error in semi-analytical model, only rectangular, dome or disk "
            "optical detectors are supported."
        << "\n";
        throw OpT0FinderException();
    }

    // calculate number of hits via geometeric acceptance
    double visibility_geo = (solid_angle_detector / (2. * geoalgo::kPI)) *
                            cathode_visibility; // 2*pi due to presence of reflective foils


    // determine correction factor, depending on PD type
    const size_t k = (theta_vis / fdelta_angulo_vis); // off-set angle bin
    double r;
    if ((opDet.orientation == 1 && fIsFlatPDCorrLat) ||
        (opDet.orientation == 2 && fIsFlatPDCorrLat))
      r = std::hypot(ScintPoint[0] - fcathode_centre[0], ScintPoint[2] - fcathode_centre[2]);
    else
      r = std::hypot(ScintPoint[1] - fcathode_centre[1], ScintPoint[2] - fcathode_centre[2]);
    double d_c = std::abs(ScintPoint[0] - plane_depth); // distance to cathode
    double border_correction = 0;
    // flat PDs
    if ((opDet.type == 0 || opDet.type == 2) && (fIsFlatPDCorr || fIsFlatPDCorrLat)) {
      // cathode/anode case
      if (opDet.orientation == 0 && fIsFlatPDCorr) {

        border_correction =
          interpolate2(fvis_distances_x_flat, fvis_distances_r_flat, fvispars_flat, d_c, r, k);
      }
      // laterals case
      else if ((opDet.orientation == 1 && fIsFlatPDCorrLat) ||
                (opDet.orientation == 2 && fIsFlatPDCorrLat)) {
        border_correction = interpolate2(fvis_distances_x_flat_lateral,
                                          fvis_distances_r_flat_lateral,
                                          fvispars_flat_lateral,
                                          d_c,
                                          r,
                                          k);
      }
      else {
        FLASH_CRITICAL()
          << "Invalid optical detector shape requested or corrections "
              "are missing - configuration error in semi-analytical model."
          << "\n";
          throw OpT0FinderException();
      }
    }
    // dome PDs
    else if (opDet.type == 1 && fIsDomePDCorr) {
      border_correction =
        interpolate2(fvis_distances_x_dome, fvis_distances_r_dome, fvispars_dome, d_c, r, k);
    }
    else {
      FLASH_CRITICAL()
        << "Error: Invalid optical detector shape requested or corrections are "
            "missing - configuration error in semi-analytical model."
        << "\n";
        throw OpT0FinderException();
    }

    // apply anode reflectivity factor
    if (AnodeMode) border_correction *= fAnodeReflectivity;

    return border_correction * visibility_geo / cosine_vis;
  }

  //......................................................................
  // Gaisser-Hillas function definition
  double SemiAnalyticalModel::Gaisser_Hillas(const double x, const double* par) const
  {
    double X_mu_0 = par[3];
    double Normalization = par[0];
    double Diff = par[1] - X_mu_0;
    double Term = std::pow((x - X_mu_0) / Diff, Diff / par[2]);
    double Exponential = std::exp((par[1] - x) / par[2]);

    return (Normalization * Term * Exponential);
  }

  //......................................................................
  // solid angle of circular aperture
  double SemiAnalyticalModel::Disk_SolidAngle(const double d, const double h, const double b) const
  {
    if (b <= 0. || d < 0. || h <= 0.) return 0.;
    const double leg2 = (b + d) * (b + d);
    const double aa = std::sqrt(h * h / (h * h + leg2));
    if (isApproximatelyZero(d)) { return 2. * geoalgo::kPI * (1. - aa); }
    double bb = 2. * std::sqrt(b * d / (h * h + leg2));
    double cc = 4. * b * d / leg2;

    if (isDefinitelyGreaterThan(d, b)) {
      try {
        return 2. * aa *
                (std::sqrt(1. - cc) * std::comp_ellint_3(bb, cc) -
                std::comp_ellint_1(bb));
      }
      catch (std::domain_error& e) {
        if (isApproximatelyEqual(d, b, 1e-9)) {
          FLASH_WARNING()
            << "Elliptic Integral in Disk_SolidAngle() given parameters "
                "outside domain."
            << "\nbb: " << bb << "\ncc: " << cc << "\nException message: " << e.what()
            << "\nRelax condition and carry on.";
          return geoalgo::kPI - 2. * aa * std::comp_ellint_1(bb);
        }
        else {
          FLASH_CRITICAL()
            << "Elliptic Integral inside Disk_SolidAngle() given parameters "
                "outside domain.\n"
            << "\nbb: " << bb << "\ncc: " << cc << "Exception message: " << e.what();
          throw OpT0FinderException();
          return 0.;
        }
      }
    }
    if (isDefinitelyLessThan(d, b)) {
      try {
        return 2. * geoalgo::kPI -
                2. * aa *
                  (std::comp_ellint_1(bb) +
                  std::sqrt(1. - cc) * std::comp_ellint_3(bb, cc));
      }
      catch (std::domain_error& e) {
        if (isApproximatelyEqual(d, b, 1e-9)) {
          FLASH_WARNING()
            << "Elliptic Integral in Disk_SolidAngle() given parameters "
                "outside domain."
            << "\nbb: " << bb << "\ncc: " << cc << "\nException message: " << e.what()
            << "\nRelax condition and carry on.";
          return geoalgo::kPI - 2. * aa * std::comp_ellint_1(bb);
        }
        else {
          FLASH_CRITICAL()
            << "Elliptic Integral inside Disk_SolidAngle() given parameters "
                "outside domain.\n"
            << "\nbb: " << bb << "\ncc: " << cc << "Exception message: " << e.what();
          throw OpT0FinderException();
          return 0.;
        }
      }
    }
    if (isApproximatelyEqual(d, b)) {
      return geoalgo::kPI - 2. * aa * std::comp_ellint_1(bb);
    }
    return 0.;
  }

  //......................................................................
  // solid angle of rectangular aperture
  double SemiAnalyticalModel::Rectangle_SolidAngle(const double a,
                                                    const double b,
                                                    const double d) const
  {
    double aa = a / (2. * d);
    double bb = b / (2. * d);
    double aux = (1. + aa * aa + bb * bb) / ((1. + aa * aa) * (1. + bb * bb));
    return 4. * fast_acos(std::sqrt(aux));
  }

  double SemiAnalyticalModel::Rectangle_SolidAngle(Dims const& o,
                                                    geoalgo::Vector_t const& v,
                                                    int OpDetOrientation) const
  {
    // v is the position of the track segment with respect to
    // the center position of the arapuca window

    // solid angle calculation depends on orientation of PD, set correct distances
    // to use
    double d1;
    double d2;
    if (OpDetOrientation == 2) {
      // lateral PD, arapuca plane fixed in z direction
      d1 = std::abs(v[0]);
      d2 = std::abs(v[2]);
    }
    else if (OpDetOrientation == 1) {
      // lateral PD, arapuca plane fixed in y direction
      d1 = std::abs(v[0]);
      d2 = std::abs(v[1]);
    }
    else {
      // anode/cathode PD, arapuca plane fixed in x direction [default]
      d1 = std::abs(v[1]);
      d2 = std::abs(v[0]);
    }
    // arapuca plane fixed in x direction
    if (isApproximatelyZero(d1) && isApproximatelyZero(v[2])) {
      return Rectangle_SolidAngle(o.h, o.w, d2);
    }
    if (isDefinitelyGreaterThan(d1, o.h * .5) &&
        isDefinitelyGreaterThan(std::abs(v[2]), o.w * .5)) {
      double A = d1 - o.h * .5;
      double B = std::abs(v[2]) - o.w * .5;
      double to_return = (Rectangle_SolidAngle(2. * (A + o.h), 2. * (B + o.w), d2) -
                          Rectangle_SolidAngle(2. * A, 2. * (B + o.w), d2) -
                          Rectangle_SolidAngle(2. * (A + o.h), 2. * B, d2) +
                          Rectangle_SolidAngle(2. * A, 2. * B, d2)) *
                          .25;
      return to_return;
    }
    if ((d1 <= o.h * .5) && (std::abs(v[2]) <= o.w * .5)) {
      double A = -d1 + o.h * .5;
      double B = -std::abs(v[2]) + o.w * .5;
      double to_return = (Rectangle_SolidAngle(2. * (o.h - A), 2. * (o.w - B), d2) +
                          Rectangle_SolidAngle(2. * A, 2. * (o.w - B), d2) +
                          Rectangle_SolidAngle(2. * (o.h - A), 2. * B, d2) +
                          Rectangle_SolidAngle(2. * A, 2. * B, d2)) *
                          .25;
      return to_return;
    }
    if (isDefinitelyGreaterThan(d1, o.h * .5) && (std::abs(v[2]) <= o.w * .5)) {
      double A = d1 - o.h * .5;
      double B = -std::abs(v[2]) + o.w * .5;
      double to_return = (Rectangle_SolidAngle(2. * (A + o.h), 2. * (o.w - B), d2) -
                          Rectangle_SolidAngle(2. * A, 2. * (o.w - B), d2) +
                          Rectangle_SolidAngle(2. * (A + o.h), 2. * B, d2) -
                          Rectangle_SolidAngle(2. * A, 2. * B, d2)) *
                          .25;
      return to_return;
    }
    if ((d1 <= o.h * .5) && isDefinitelyGreaterThan(std::abs(v[2]), o.w * .5)) {
      double A = -d1 + o.h * .5;
      double B = std::abs(v[2]) - o.w * .5;
      double to_return = (Rectangle_SolidAngle(2. * (o.h - A), 2. * (B + o.w), d2) -
                          Rectangle_SolidAngle(2. * (o.h - A), 2. * B, d2) +
                          Rectangle_SolidAngle(2. * A, 2. * (B + o.w), d2) -
                          Rectangle_SolidAngle(2. * A, 2. * B, d2)) *
                          .25;
      return to_return;
    }

    return 0.;
  }

  //......................................................................
  // solid angle of dome aperture
  double SemiAnalyticalModel::Omega_Dome_Model(const double distance, const double theta) const
  {
    // this function calculates the solid angle of a semi-sphere of radius b,
    // as a correction to the analytic formula of the on-axix solid angle,
    // as we move off-axis an angle theta. We have used 9-angular bins
    // with delta_theta width.

    // par0 = Radius correction close
    // par1 = Radius correction far
    // par2 = breaking distance betwween "close" and "far"

    constexpr double par0[9] = {0., 0., 0., 0., 0., 0.597542, 1.00872, 1.46993, 2.04221};
    constexpr double par1[9] = {
      0., 0., 0.19569, 0.300449, 0.555598, 0.854939, 1.39166, 2.19141, 2.57732};
    const double delta_theta = 10.; // TODO: should this be fdelta_angulo_vuv?
    int j = int(theta / delta_theta);
    // PMT radius
    const double b = fradius; // cm
    // distance form which the model parameters break (empirical value)
    const double d_break = 5 * b; // par2

    if (distance >= d_break) {
      double R_apparent_far = b - par1[j];
      double ratio_square = (R_apparent_far * R_apparent_far) / (distance * distance);
      return (2 * geoalgo::kPI * (1 - std::sqrt(1 - ratio_square)));
    }
    else {
      double R_apparent_close = b - par0[j];
      double ratio_square = (R_apparent_close * R_apparent_close) / (distance * distance);
      return (2 * geoalgo::kPI * (1 - std::sqrt(1 - ratio_square)));
    }
  }

  //......................................................................
  // checks photo-detector is in same TPC/argon volume as scintillation
  bool SemiAnalyticalModel::isOpDetInSameTPC(geoalgo::Point_t const& ScintPoint,
                                              geoalgo::Point_t const& OpDetPoint) const
  {
    // method working for SBND, uBooNE, DUNE-HD 1x2x6 and DUNE-VD 1x8x6
    // will need to be replaced to work in full DUNE geometry, ICARUS geometry
    // check x coordinate has same sign or is close to zero, otherwise return
    // false

    if (((ScintPoint[0] < 0.) != (OpDetPoint[0] < 0.)) && std::abs(OpDetPoint[0]) > 10. &&
        fNTPC == 2) { 
      return false;
    }
    return true;
  }

  std::vector<SemiAnalyticalModel::OpticalDetector> SemiAnalyticalModel::opticalDetectors() const
  {
    std::vector<geoalgo::Point_t> opDetCenters = DetectorSpecs::GetME().PMTPositions();
    std::vector<SemiAnalyticalModel::OpticalDetector> opticalDetector;
    for (size_t ch = 0; ch < opDetCenters.size(); ch++) {
      geoalgo::Point_t center(opDetCenters[ch][0], opDetCenters[ch][1], opDetCenters[ch][2]);
      //Continue if channel is NOT in _channel_mask
      // if (std::find(_channel_mask.begin(), _channel_mask.end(), ch) == _channel_mask.end()) {
      //   FLASH_DEBUG() << "SKIPPED Optical Detector " << ch << " center: " << center[0] << " " << center[1] << " " << center[2] << std::endl;
      //   continue;
      // }
      assert(center.size()==3);
      // Check if it's rectangular or spherical
      if (std::find(fspherical_ids.begin(), fspherical_ids.end(), ch) != fspherical_ids.end()) {
        opticalDetector.emplace_back(
          SemiAnalyticalModel::OpticalDetector{-1,-1, center, fspherical_type, fspherical_orientation});
      }
      else if (std::find(frectengular_ids.begin(), frectengular_ids.end(), ch) != frectengular_ids.end()) {
        opticalDetector.emplace_back(
          SemiAnalyticalModel::OpticalDetector{frectengular_height, frectengular_width, center, frectengular_type, frectengular_orientation});
      }
      else {
        FLASH_CRITICAL() << "Unknown optical detector type for id: " << ch << std::endl;
        throw OpT0FinderException();
      }
      FLASH_DEBUG() << "Optical Detector " << ch << " center: " << center[0] << " " << center[1] << " " << center[2] << std::endl;
      //ch++;
    }
    return opticalDetector;
  }
  void SemiAnalyticalModel::Print() const {
    std::cout << "-------------------------------------------------" << std::endl;
    std::cout << "SemiAnalyticalModel Configuration:" << std::endl;

    std::cout << "Active Volume Center: (" << fActiveVolume.Center()[0] << ", "
              << fActiveVolume.Center()[1] << ", " << fActiveVolume.Center()[2] << ")" << std::endl;
    std::cout << "Anode Center: (" << fanode_centre[0] << ", " << fanode_centre[1] << ", " << fanode_centre[2] << ")" << std::endl;
    std::cout << "Cathode Center: (" << fcathode_centre[0] << ", " << fcathode_centre[1] << ", " << fcathode_centre[2] << ")" << std::endl;

    std::cout << "Do Reflected Light: " << std::boolalpha << fDoReflectedLight << std::endl;
    std::cout << "Include Anode Reflections: " << fIncludeAnodeReflections << std::endl;
    std::cout << "Use Xenon Absorption: " << fUseXeAbsorption << std::endl;

    std::cout << "VUV Absorption Length: " << fvuv_absorption_length << " cm" << std::endl;
    std::cout << "Spherical Type: " << fspherical_type << std::endl;
    std::cout << "Spherical Orientation: " << fspherical_orientation << std::endl;
    std::cout << "Spherical IDs: ";
    for (auto id : fspherical_ids) std::cout << id << " ";
    std::cout << std::endl;

    std::cout << "Rectangular Type: " << frectengular_type << std::endl;
    std::cout << "Rectangular Orientation: " << frectengular_orientation << std::endl;
    std::cout << "Rectangular Height: " << frectengular_height << std::endl;
    std::cout << "Rectangular Width: " << frectengular_width << std::endl;
    std::cout << "Rectangular IDs: ";
    for (auto id : frectengular_ids) std::cout << id << " ";
    std::cout << std::endl;

    std::cout << "Number of Optical Detectors: " << fNOpDets << std::endl;

    std::cout << "Gaisser-Hillas VUV Parameters:" << std::endl;
    if (fIsFlatPDCorr) {
        std::cout << "  Flat Correction:" << std::endl;
        for (const auto& vec : fGHvuvpars_flat) {
            for (double val : vec) {
                std::cout << "    " << val << std::endl;
            }
        }
    }
    if (fIsDomePDCorr) {
        std::cout << "  Dome Correction:" << std::endl;
        for (const auto& vec : fGHvuvpars_dome) {
            for (double val : vec) {
                std::cout << "    " << val << std::endl;
            }
        }
    }
    if (fIsFlatPDCorrLat) {
        std::cout << "  Flat Lateral Correction:" << std::endl;
        for (const auto& vec : fGHvuvpars_flat_lateral) {
            for (double val : vec) {
                std::cout << "    " << val << std::endl;
            }
        }
    }
    //Print absoprtion spectrum
    for (auto elem : abs_length_spectrum) {
      std::cout
        << "Wavelength: " << elem.first << " cm, Absorption Energy: " << elem.second << " eV"
        << std::endl;
    }

    //Print QE of optical detectors
    std::cout << "Optical Detector Quantum Efficiencies (_qe_v): " << std::endl;
    for (double qe : _qe_v) {
      std::cout << qe << " ";
    }
    std::cout << std::endl;
    std::cout << "Optical Detector reflected Quantum Efficiencies (_qe_refl_v): " << std::endl;
    for (double qe : _qe_refl_v) {
      std::cout << qe << " ";
    }
    std::cout << std::endl;

    std::cout << "End of SemiAnalyticalModel Configuration." << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;
  }

  void SemiAnalyticalModel::printVUVParameters() const {
    std::cout << "FlatPDCorr: " << (fIsFlatPDCorr ? "true" : "false") << std::endl;
    std::cout << "FlatPDCorrLat: " << (fIsFlatPDCorrLat ? "true" : "false") << std::endl;
    std::cout << "DomePDCorr: " << (fIsDomePDCorr ? "true" : "false") << std::endl;
    std::cout << "Delta Angulo VUV: " << fdelta_angulo_vuv << std::endl;
    std::cout << "PMT Radius: " << fradius << std::endl;
    std::cout << "ApplyFieldCageTransparency: " << (fApplyFieldCageTransparency ? "true" : "false") << std::endl;
    std::cout << "FieldCageTransparencyLateral: " << fFieldCageTransparencyLateral << std::endl;
    std::cout << "FieldCageTransparencyCathode: " << fFieldCageTransparencyCathode << std::endl;
    std::cout << "VUV absorption length: " << fvuv_absorption_length << std::endl;
    if (fIsFlatPDCorr) {
        std::cout << "GH Parameters Flat: " << std::endl;
        for (const auto& vec : fGHvuvpars_flat) {
            for (double val : vec) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "Border Correction Angle Flat: ";
        for (double val : fborder_corr_angulo_flat) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        std::cout << "Border Correction Flat: " << std::endl;
        for (const auto& vec : fborder_corr_flat) {
            for (double val : vec) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }

    if (fIsFlatPDCorrLat) {
        std::cout << "GH Parameters Flat Lateral: " << std::endl;
        for (const auto& vec : fGHvuvpars_flat_lateral) {
            for (double val : vec) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "Border Correction Angle Flat Lateral: ";
        for (double val : fborder_corr_angulo_flat_lateral) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        std::cout << "Border Correction Flat Lateral: " << std::endl;
        for (const auto& vec : fborder_corr_flat_lateral) {
            for (double val : vec) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }

    if (fIsDomePDCorr) {
        std::cout << "GH Parameters Dome: " << std::endl;
        for (const auto& vec : fGHvuvpars_dome) {
            for (double val : vec) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "Border Correction Angle Dome: ";
        for (double val : fborder_corr_angulo_dome) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        std::cout << "Border Correction Dome: " << std::endl;
        for (const auto& vec : fborder_corr_dome) {
            for (double val : vec) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }
  }

  void SemiAnalyticalModel::printVISParameters() const {
    std::cout << "Delta Angulo VIS: " << fdelta_angulo_vis << std::endl;

    if (fIsFlatPDCorr) {
        std::cout << "VIS Distances X Flat: ";
        for (double val : fvis_distances_x_flat) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        std::cout << "VIS Distances R Flat: ";
        for (double val : fvis_distances_r_flat) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        std::cout << "VIS Corrections Flat: " << std::endl;
        for (const auto& layer : fvispars_flat) {
            for (const auto& row : layer) {
                for (double val : row) {
                    std::cout << val << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "------" << std::endl;
        }
    }

    if (fIsFlatPDCorrLat) {
        std::cout << "VIS Distances X Flat Lateral: ";
        for (double val : fvis_distances_x_flat_lateral) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        std::cout << "VIS Distances R Flat Lateral: ";
        for (double val : fvis_distances_r_flat_lateral) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        std::cout << "VIS Corrections Flat Lateral: " << std::endl;
        for (const auto& layer : fvispars_flat_lateral) {
            for (const auto& row : layer) {
                for (double val : row) {
                    std::cout << val << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "------" << std::endl;
        }
    }

    if (fIsDomePDCorr) {
        std::cout << "VIS Distances X Dome: ";
        for (double val : fvis_distances_x_dome) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        std::cout << "VIS Distances R Dome: ";
        for (double val : fvis_distances_r_dome) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        std::cout << "VIS Corrections Dome: " << std::endl;
        for (const auto& layer : fvispars_dome) {
            for (const auto& row : layer) {
                for (double val : row) {
                    std::cout << val << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "------" << std::endl;
        }
    }

    std::cout << "Cathode Plane Height: " << fcathode_plane.h << std::endl;
    std::cout << "Cathode Plane Width: " << fcathode_plane.w << std::endl;
    std::cout << "Plane Depth: " << fplane_depth << std::endl;
    std::cout << "Anode Reflectivity: " << fAnodeReflectivity << std::endl;
  }
} // namespace flashmatch
