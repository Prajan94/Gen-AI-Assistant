import { Component } from '@angular/core';
import { CommonServiceService } from '../common-service.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-synthetic-data-gen',
  templateUrl: './synthetic-data-gen.component.html',
  styleUrl: './synthetic-data-gen.component.scss'
})
export class SyntheticDataGenComponent {
  response: any;
  aiAuery: any;
  course: any;
  rows: any;
  errMsg: any;
  hideBotMessage = false;
  isLoading = false;
  aiMessageArray: any = [];
  humanMessageArray:any = [];
  constructor(
    private commonService : CommonServiceService, private route: Router) { }
    askAi() {
      this.errMsg = ""
      if (this.rows) {
          this.hideBotMessage = true;
          this.isLoading = true;
          this.aiMessageArray.push(this.aiAuery);
          this.commonService.genSynData(this.aiAuery, this.rows).subscribe(item => {
          this.response = JSON.stringify(item);
          this.humanMessageArray.push(this.response);
          this.isLoading = false;
        }, ((error: any) => {
          this.isLoading = false;
          console.log("Error from flask server --->" + error);
          this.response = "Network Error occured, please try again";
        }))
      } else {
        this.errMsg = "Please mention rows"
      }

  }
  gotoHome() {
    this.route.navigateByUrl("home");
  }
}
